# Multilingual SMS Spam Detection via PCA-Based Embedding Fusion

This repository contains the executable research code for the
leakage-controlled experiments reported in the revised manuscript
"Multilingual SMS Spam Detection via PCA-Based Embedding Fusion."
The principal Table 7 neural evaluation uses five disjoint source-group test
folds and five training seeds per fold. Sparse controls use the same outer
folds with fixed estimator settings. Earlier message-level and fixed-split
workflows are retained only for comparison.

The workflow covers source-group splitting, conventional text baselines,
static embedding fusion, dimensionality-reduction controls, fine-tuned mBERT,
statistical analysis, and resource profiling. Translated versions of one
source message are assigned to the same data partition.

## Repository contents

- `analysis/revision_audit.py`: preprocessing, message-level-versus-group split audit,
  and reference baseline analysis.
- `analysis/grouped_baseline_benchmark.py`: grouped conventional baselines.
- `analysis/grouped_embedding_fusion_experiment.py`: static embedding fusion,
  PCA/random-projection controls, CNN training, evaluation, and profiling.
- `analysis/grouped_qwen_embedding_fusion_experiment.py`: grouped Qwen2.5 and
  mBERT/Qwen2.5 centered-PCA extension with NF4 offline extraction.
- `analysis/grouped_qwen_pca_sweep_experiment.py`: validation-only PCA/RP width
  selection, corrected zero-padding treatment, matched five-seed controls,
  source-group bootstrap, surface-perturbation tests, and trade-off plots.
- `analysis/grouped_mbert_qwen_char_tfidf_concat_pca_cnn.py`: token-aligned
  character TF-IDF extension, joint mBERT/Qwen2.5/character-feature PCA width
  selection, matched five-seed CNN evaluation, and resource profiling.
- `analysis/evaluate_all_pca_dimensions_test.py`: post-selection five-seed test
  evaluation of every validation-tested PCA width for the embedding-only and
  character-feature fusion configurations.
- `analysis/evaluate_distil_mbert_pca1024_test.py`: matched five-seed test
  evaluation of DistilBERT/mBERT Concat+PCA-1024.
- `analysis/evaluate_tfidf_svd1024_test.py`: fixed-split 1,024-dimensional
  Truncated-SVD controls for the retained TF-IDF/classifier variations.
- `analysis/evaluate_tfidf_pca1024_test.py`: matched centered randomized
  PCA-1024 controls for the same character/word TF-IDF features and classifiers.
- `analysis/audit_embedding_extraction.py`: native-tokenizer embedding audit.
- `analysis/finetune_mbert_grouped.py`: grouped fine-tuned mBERT experiment.
- `analysis/run_cv5_multiseed_all_static.py`: complete five-fold by five-seed
  static-vector CNN evaluation.
- `analysis/summarize_multiseed_static.py`: 25-run summaries, validation-based
  width selection, fold-clustered comparisons, and source-group bootstrap.
- `analysis/run_revision_gap_experiments.py`: matched averaging, trainable
  source gating, and late-fusion controls for mBERT/Qwen2.5 and
  Llama-2/Qwen2.5, plus a separate DistilBERT/mBERT vocabulary-scope diagnostic.
- `analysis/summarize_revision_gap_experiments.py`: cumulative explained
  variance, paired source-group intervals, and matched confusion matrices.
- `analysis/test_revision_gap_experiments.py`: checks for source alignment,
  gate gradients, and equivalence of training-time gating and frozen lookup.
- `analysis/run_external_validation_multiseed.py`: 25-model external-corpus
  and perturbation evaluation.
- `analysis/summarize_transformer_multiseed.py` and
  `analysis/compare_mbert_multiseed.py`: repeated-run fine-tuned-mBERT summary
  and paired source-group comparisons.
- `analysis/summarize_*.py`: statistical and language-level summaries.
- `analysis/make_source_group_folds.py`, `prepare_cv5_embedding_matrices.py`,
  `run_cv5_fusion_core.py`, and `summarize_source_group_cv5.py`: the five-fold
  source-group workflow used for the principal Table 7 estimates.
- `analysis/prepare_cv5_llama_embeddings.py`, `run_cv5_llama_fusion.py`, and
  `run_cv5_triple_fusion.py`: matched Llama-2/Qwen2.5 and
  mBERT/Qwen2.5/Llama-2 input-table controls. Their summaries and paired
  comparisons are produced by `summarize_cv5_llama_fusion.py` and
  `summarize_cv5_triple_fusion.py`.
- `REPRODUCE_CV5.md`: commands and partition details for the five-fold results.
- `requirements.txt`: verified Python dependencies.
- `results/final_multiseed/`: final aggregate numerical results and SHA-256
  checksums. No message text or per-message prediction is included.
- `results/revision_gap_experiments/`: the additional matched-control metrics,
  paired intervals, explained-variance summaries, and confusion counts, with
  SHA-256 checksums. The full-vocabulary experiment is diagnostic only and
  is not pooled with the training-only primary results.

The repository intentionally does not redistribute SMS texts or per-message
predictions. The scripts write all generated splits, predictions, summaries,
and audit files to user-selected output directories.

## Environment

The verified environment used Python 3.11.9 with the package versions pinned
in `requirements.txt`. The neural experiments were run on an NVIDIA GeForce
RTX 5060 Laptop GPU. Public pretrained model weights are downloaded from their
providers on first execution.

The model identifiers are:

- `distilbert-base-multilingual-cased`
- `bert-base-multilingual-uncased`
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-2-7b` (licensed, gated checkpoint; weights not redistributed)

## Data preparation

Obtain the public
[SMS Spam Multilingual Collection Dataset](https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset)
from its provider and save the wide-format table as a Parquet file. The input
must contain `group_id`, `labels`, and the multilingual text columns (`text`
and columns beginning with `text_`). See `DATA_LICENSE.md` for attribution and
licensing information. The bilingual Indonesian-English corpus is not used by
these source-group scripts and is not distributed here.

## Reproduction

Create an isolated Python environment and install the dependencies. Follow
`REPRODUCE_CV5.md` for the principal five-fold by five-seed Table 7 and
fine-tuned-mBERT experiments. The commands below reproduce the earlier
fixed-split experiments. Replace
`<dataset.parquet>` with the prepared multilingual dataset path.

```powershell
python -m pip install -r requirements.txt
python analysis/revision_audit.py --dataset <dataset.parquet> --output-dir results/classical
python analysis/grouped_baseline_benchmark.py --dataset <dataset.parquet> --output-dir results/classical
python analysis/grouped_embedding_fusion_experiment.py --dataset <dataset.parquet> --output-dir results/grouped_embedding_fusion --batch-size 1024 --embedding-batch-size 256 --max-epochs 30 --max-length 128
python analysis/summarize_fusion_experiment.py --experiment-dir results/grouped_embedding_fusion --bootstrap-repetitions 1000
python analysis/audit_embedding_extraction.py --experiment-dir results/grouped_embedding_fusion --batch-size 256
python analysis/grouped_qwen_embedding_fusion_experiment.py --dataset <dataset.parquet> --model-path models/Qwen2.5-7B-Instruct --mbert-matrix results/grouped_embedding_fusion/embedding_bert-base-multilingual-uncased.npy --reference-vocabulary results/grouped_embedding_fusion/training_vocabulary.csv --output-dir results/grouped_qwen_embedding_fusion --batch-size 1024 --embedding-batch-size 16 --max-epochs 30 --max-length 128
python analysis/summarize_qwen_fusion_experiment.py --new-dir results/grouped_qwen_embedding_fusion --old-dir results/grouped_embedding_fusion --bootstrap-repetitions 5000
python analysis/grouped_qwen_pca_sweep_experiment.py --dataset <dataset.parquet> --mbert-matrix results/grouped_embedding_fusion/embedding_bert-base-multilingual-uncased.npy --qwen-matrix results/grouped_qwen_embedding_fusion/embedding_qwen2.5-7b-instruct_nf4.npy --reference-vocabulary results/grouped_embedding_fusion/training_vocabulary.csv --output-dir results/grouped_qwen_pca_sweep --dims 768 1024 1536 2048 --physical-batch-size 256 --effective-batch-size 1024 --max-epochs 30 --max-length 128
python analysis/grouped_mbert_qwen_char_tfidf_concat_pca_cnn.py --dataset <dataset.parquet> --embedding-concat results/grouped_qwen_pca_sweep/embedding_mbert_qwen_standardized_concat.npy --reference-vocabulary results/grouped_embedding_fusion/training_vocabulary.csv --output-dir results/grouped_mbert_qwen_char_tfidf_concat_pca_cnn --char-dimension 1024 --dims 768 1024 1536 2048 --physical-batch-size 256 --effective-batch-size 1024 --max-epochs 30 --max-length 128
python analysis/evaluate_all_pca_dimensions_test.py --dataset <dataset.parquet> --reference-vocabulary results/grouped_embedding_fusion/training_vocabulary.csv --qwen-pca-matrix results/grouped_qwen_pca_sweep/embedding_mbert_qwen_pca2048_max.npy --qwen-experiment-dir results/grouped_qwen_pca_sweep --char-pca-matrix results/grouped_mbert_qwen_char_tfidf_concat_pca_cnn/embedding_mbert_qwen_char_concat_pca2048_max.npy --char-experiment-dir results/grouped_mbert_qwen_char_tfidf_concat_pca_cnn --output-dir results/all_pca_dimensions_test --dims 768 1024 1536 2048 --physical-batch-size 256 --effective-batch-size 1024 --max-epochs 30 --max-length 128
python analysis/evaluate_distil_mbert_pca1024_test.py --dataset <dataset.parquet> --source-dir results/grouped_embedding_fusion --output-dir results/distil_mbert_pca1024_test --batch-size 1024 --max-epochs 30 --max-length 128
python analysis/evaluate_tfidf_svd1024_test.py --dataset <dataset.parquet> --output-dir results/tfidf_svd1024_test --svd-iterations 4
python analysis/evaluate_tfidf_pca1024_test.py --dataset <dataset.parquet> --output-dir results/tfidf_pca1024_test --power-iterations 4
python analysis/finetune_mbert_grouped.py --dataset <dataset.parquet> --output-dir results/finetuned_mbert_grouped --batch-size 32 --max-length 64 --max-epochs 4 --early-stopping-patience 2 --threshold-mode validation_mcc --seeds 13 21 42 87 101
python analysis/summarize_finetuned_mbert.py --audit-dir results --bootstrap-repetitions 1000
```

The commands above reproduce the earlier fixed-partition audit; the manuscript's
primary results use the five-fold by five-seed workflow in `REPRODUCE_CV5.md`.
Download the official Qwen2.5
checkpoint to the path passed through `--model-path`; it is loaded in NF4 only
for offline vocabulary extraction. The PCA sweep excludes the padding entry
from fitted transformations and resets it to zero afterward. Conventional
baselines and fine-tuned mBERT use the same five source-group folds and training
seeds 13, 21, 42, 87, and 101.

## Scope

This code reproduces the source-group experiments added during revision. The
historical message-level tables are retained in the manuscript only as explicitly
labeled exploratory results and are not the primary evidence of the revised
study.

## Final repeated-run results

The final static suite contains 25 held-out runs for each configuration. Mean
MCC is 0.7906 ± 0.0215 for validation-selected mBERT/Qwen2.5 PCA and
0.7922 ± 0.0161 for validation-selected mBERT/Qwen2.5/character fusion; their
paired ensemble difference is not separated from zero. Fine-tuned mBERT obtains
0.8853 ± 0.0160 MCC. External results are reported for ExAIS, TurkishSMS,
YouTube Spam Collection, and SpamAssassin Public Corpus without external-label
adaptation.

The matched fusion controls add 25 evaluations per method and encoder pair.
For mBERT/Qwen2.5, averaging, source gating, and late fusion yield mean MCC
0.7462, 0.7658, and 0.7855; selected PCA yields 0.7906. Its paired ensemble
contrast with late fusion includes zero. For Llama-2/Qwen2.5, late fusion
yields 0.8173 compared with 0.7816 for selected PCA, with a negative
PCA-minus-late-fusion interval. The separate DistilBERT/mBERT PCA-1024
diagnostic yields 0.7155 with full-vocabulary fitting versus 0.7399 with
training-only fitting. Full results, uncertainty, and scope are retained in
`results/revision_gap_experiments/`.

## Citation

Use the version-specific Zenodo DOI stated in the latest GitHub release. Older
DOIs identify earlier experimental snapshots and should not be used for the
final five-fold by five-seed results.
