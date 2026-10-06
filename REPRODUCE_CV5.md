# Five-fold source-group evaluation

This workflow reproduces the primary experiments in the revised manuscript.
It uses five disjoint source-group test folds and five training seeds per fold
(13, 21, 42, 87, and 101). A stratified 20% of the remaining source groups is
used for validation in each fold. Text features, scaling statistics, projection
operators, model selection, and decision thresholds are fitted on training or
validation data only; the test fold is evaluated once per fitted model.

## Prerequisites

Install `requirements.txt`, obtain the public multilingual dataset, and prepare
the frozen embedding matrices described in the main `README.md`. The following
intermediate files are required but are not stored in Git because of their size:

- `results/grouped_embedding_fusion/training_vocabulary.csv`
- `results/grouped_embedding_fusion/embedding_bert-base-multilingual-uncased.npy`
- `results/grouped_embedding_fusion/embedding_distilbert-base-multilingual-cased.npy`
- `results/grouped_qwen_embedding_fusion/embedding_qwen2.5-7b-instruct_nf4.npy`

The commands below are intended for PowerShell from the repository root.

```powershell
$data = 'data/sms_spam_multilingual.parquet'
$cv = 'results/source_group_cv5'
$runs = 'results/multiseed_all'
$seeds = 13, 21, 42, 87, 101

python analysis/make_source_group_folds.py --dataset $data --output-dir $cv
python analysis/prepare_cv5_embedding_matrices.py --dataset $data --fold-dir $cv --base-vocabulary results/grouped_embedding_fusion/training_vocabulary.csv --base-mbert results/grouped_embedding_fusion/embedding_bert-base-multilingual-uncased.npy --base-distil results/grouped_embedding_fusion/embedding_distilbert-base-multilingual-cased.npy --base-qwen results/grouped_qwen_embedding_fusion/embedding_qwen2.5-7b-instruct_nf4.npy --qwen-model models/Qwen2.5-7B-Instruct
python analysis/grouped_baseline_benchmark.py --dataset $data --fold-dir $cv --output-dir "$cv/classical"
python analysis/run_cv5_multiseed_all_static.py --dataset $data --fold-dir $cv --output-dir $runs --seeds $seeds
python analysis/summarize_multiseed_static.py --runs "$runs/static_multiseed_runs.csv" --dataset $data --fold-dir $cv --prediction-dir "$runs/predictions" --output-dir "$runs/summary" --bootstrap-repetitions 2000
python analysis/plot_cv5_tradeoff.py --fold-dir $cv --runs "$runs/static_multiseed_runs.csv" --output-dir "$runs/figures"
```

The static suite includes PCA and no-PCA variants within every reported feature
family, validation-selected PCA widths, a 1,024-dimensional random-projection
control, sparse character and word TF-IDF baselines, Llama-2/Qwen2.5 fusion,
and three-encoder fusion. PCA dimensions are selected by validation MCC, never
by test performance.

## Matched fusion and vocabulary controls

Run the existing static suite first. The following adds averaging and
input-dependent MoE for mBERT/Qwen2.5 and Llama-2/Qwen2.5, and computes
equal-weight late fusion from the matched branch predictions. It also compares
training-only and full-corpus vocabulary fitting for DistilBERT/mBERT PCA-1024
on identical folds, training repetitions, and classifier settings. The
full-vocabulary configuration is a sensitivity diagnostic, not a primary
leakage-controlled result.

```powershell
$gaps = 'results/revision_gap_experiments'
python analysis/run_revision_gap_experiments.py --dataset $data --fold-dir $cv --static-dir $runs --output-dir $gaps --mode all
python analysis/test_revision_gap_experiments.py
python analysis/summarize_revision_gap_experiments.py --fold-dir $cv --static-dir $runs --output-dir $gaps --figure-output "$gaps/figures/grouped_confusion.pdf"
```

The runner checkpoints each completed evaluation and can resume with the same
arguments. Large aligned fusion matrices are held in memory rather than written
to disk. Frozen encoder weights and intermediate embedding matrices remain
local. The small PCA diagnostic cache and processed messages are excluded from
the public numerical archive. Public results include mean/standard-deviation
summaries, fold-clustered comparisons, paired source-group bootstrap intervals,
explained variance for all fitted combinations, and matched confusion counts.
Late-fusion cost fields sum the measured independent branch workloads; they do
not measure aggregation overhead or concurrent execution latency.

## Fine-tuned mBERT

Fine-tuned mBERT is evaluated on the same five outer test folds and five seeds.
Early stopping and the probability threshold are selected from the validation
partition independently for each run.

```powershell
$transformers = 'results/transformer_multiseed'
for ($fold = 1; $fold -le 5; $fold++) {
    python analysis/finetune_mbert_grouped.py --dataset $data --fold-file "$cv/fold_$fold.npz" --output-dir "$transformers/fold_$fold/mbert" --batch-size 32 --max-length 64 --max-epochs 4 --early-stopping-patience 2 --threshold-mode validation_mcc --seeds $seeds
}
python analysis/summarize_transformer_multiseed.py --dataset $data --fold-dir $cv --transformer-dir $transformers --output-dir "$transformers/summary" --bootstrap-repetitions 2000
```

## External-corpus evaluation

External corpora are evaluated without fitting to external labels. Supply the
audited corpus files as `name=path` entries and reuse the 25 development models.

```powershell
python analysis/run_external_validation_multiseed.py --development $data --fold-dir $cv --static-runs "$runs/all_test_fold_runs.csv" --external ExAIS=data/external/exais.csv TurkishSMS=data/external/turkish_sms.csv YouTube=data/external/youtube_spam.csv SpamAssassin=data/external/spamassassin.csv --seed42-model-dir results/source_group_cv5 --output-dir results/external_validation_multiseed --seeds $seeds
```

## Llama-2 checkpoint

The Llama-2 rows use frozen averages of the raw input-token embedding table from
`meta-llama/Llama-2-7b`, not full-model contextual outputs. Obtain the official
gated checkpoint under its license and place `consolidated.00.pth`,
`tokenizer.model`, and `params.json` in `models/Llama-2-7b`. Model weights and
authentication credentials are not distributed in this repository.

The released aggregate summaries contain no SMS text, per-message predictions,
credentials, or model weights. Paired bootstrap intervals resample source groups
from pooled out-of-fold predictions and therefore preserve the evaluation unit.
