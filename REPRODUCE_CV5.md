# Five-fold source-group evaluation

This workflow reproduces the matched-fold results in the revised manuscript.
It uses five stratified outer folds of retained source-message identifiers.
Each test source group appears in exactly one fold. Standalone mBERT is
evaluated with and without PCA at 256 and 512 dimensions; the other static
and sparse feature families also retain PCA and no-PCA controls. A stratified 20% of the
remaining groups forms that fold's validation partition. All fitted text
features, scaling statistics, and projections use training data only. Model
training seed 42 is held fixed across folds.

## Prerequisites

Install `requirements.txt` and obtain the public multilingual dataset and
pretrained checkpoints listed in the main `README.md`. The base extraction
commands there first create these intermediate files (large matrices are not
stored in Git):

- `results/grouped_embedding_fusion/training_vocabulary.csv`
- `results/grouped_embedding_fusion/embedding_bert-base-multilingual-uncased.npy`
- `results/grouped_embedding_fusion/embedding_distilbert-base-multilingual-cased.npy`
- `results/grouped_qwen_embedding_fusion/embedding_qwen2.5-7b-instruct_nf4.npy`

The Qwen2.5 path below must point to the downloaded official checkpoint.
Commands are shown for PowerShell from the repository root.

```powershell
$data = 'data/sms_spam_multilingual.parquet'
$cv = 'results/source_group_cv5'
python analysis/make_source_group_folds.py --dataset $data --output-dir $cv
python analysis/prepare_cv5_embedding_matrices.py --dataset $data --fold-dir $cv --base-vocabulary results/grouped_embedding_fusion/training_vocabulary.csv --base-mbert results/grouped_embedding_fusion/embedding_bert-base-multilingual-uncased.npy --base-distil results/grouped_embedding_fusion/embedding_distilbert-base-multilingual-cased.npy --base-qwen results/grouped_qwen_embedding_fusion/embedding_qwen2.5-7b-instruct_nf4.npy --qwen-model models/Qwen2.5-7B-Instruct
python analysis/grouped_baseline_benchmark.py --dataset $data --fold-dir $cv --output-dir "$cv/classical"
for ($fold = 1; $fold -le 5; $fold++) {
    python analysis/run_cv5_fusion_core.py --dataset $data --fold-dir $cv --fold $fold --end-index 4
    python analysis/run_cv5_fusion_core.py --dataset $data --fold-dir $cv --fold $fold --only-index 5
    python analysis/run_cv5_fusion_core.py --dataset $data --fold-dir $cv --fold $fold --start-index 6 --end-index 10
    python analysis/run_cv5_fusion_core.py --dataset $data --fold-dir $cv --fold $fold --only-index 11
    python analysis/run_cv5_fusion_core.py --dataset $data --fold-dir $cv --fold $fold --start-index 12
    python analysis/evaluate_tfidf_pca1024_test.py --dataset $data --fold-file "$cv/fold_$fold.npz" --seeds 42 --output-dir "$cv/fold_$fold/sparse_pca1024"
    python analysis/evaluate_tfidf_svd1024_test.py --dataset $data --fold-file "$cv/fold_$fold.npz" --seeds 42 --output-dir "$cv/fold_$fold/sparse_svd1024"
    python analysis/finetune_mbert_grouped.py --dataset $data --fold-file "$cv/fold_$fold.npz" --output-dir "$cv/fold_$fold/finetuned_mbert" --batch-size 32 --max-length 64 --max-epochs 1 --seeds 42
}
python analysis/summarize_source_group_cv5.py --fold-dir $cv
python analysis/paired_cv5_bootstrap.py --dataset $data --fold-dir $cv --repetitions 2000
python analysis/paired_sparse_projection_cv5.py --dataset $data --fold-dir $cv --repetitions 2000
python analysis/paired_static_projection_cv5.py --dataset $data --fold-dir $cv --repetitions 2000
```

## Llama-2 and three-encoder extensions

The Llama-2 rows use frozen averages of the raw input-token embedding table
from `meta-llama/Llama-2-7b`, not full-model contextual outputs and not the
precomputed Llama vectors from the original message-level study. Obtain the
checkpoint through its official gated distribution under its license and place
`consolidated.00.pth`, `tokenizer.model`, and `params.json` in
`models/Llama-2-7b`. Model weights and authentication credentials are not
included in this archive. The existing mBERT and Qwen fold matrices from the
commands above are also required.

```powershell
python analysis/prepare_cv5_llama_embeddings.py --fold-dir $cv --model-path models/Llama-2-7b --batch-size 256
for ($fold = 1; $fold -le 5; $fold++) {
    python analysis/run_cv5_llama_fusion.py --dataset $data --fold-dir $cv --fold $fold
    python analysis/run_cv5_triple_fusion.py --dataset $data --fold-dir $cv --fold $fold
}
python analysis/summarize_cv5_llama_fusion.py --dataset $data --fold-dir $cv --repetitions 2000
python analysis/summarize_cv5_triple_fusion.py --dataset $data --fold-dir $cv --repetitions 2000
```

Both extensions use the saved five disjoint source-group test folds and one
training seed per fold. Candidate PCA widths are chosen using only validation
MCC. The manuscript's supplementary reproducibility archive includes fold
runs, held-out predictions, summaries, and paired source-group bootstrap
results. This code repository does not redistribute SMS texts, predictions,
or large regenerated embedding matrices.

The index ranges only limit per-process memory; completed configurations are
skipped on restart. `all_test_fold_summary.csv` reports means and sample
standard deviations over held-out folds. Paired intervals resample source
groups from pooled out-of-fold predictions, conditional on the trained models;
they do not estimate variability across retraining seeds.
