# Multilingual SMS Spam Detection via PCA-Based Embedding Fusion

This repository contains the executable research code for the
leakage-controlled experiments reported in the revised manuscript
"Multilingual SMS Spam Detection via PCA-Based Embedding Fusion."

The archival release is available at
[https://doi.org/10.5281/zenodo.22788881](https://doi.org/10.5281/zenodo.22788881)
(release `v1.0.5`, commit `ce8003a`).

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
- `analysis/audit_embedding_extraction.py`: native-tokenizer embedding audit.
- `analysis/finetune_mbert_grouped.py`: grouped fine-tuned mBERT experiment.
- `analysis/summarize_*.py`: statistical and language-level summaries.
- `requirements.txt`: verified Python dependencies.

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

## Data preparation

Obtain the public
[SMS Spam Multilingual Collection Dataset](https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset)
from its provider and save the wide-format table as a Parquet file. The input
must contain `group_id`, `labels`, and the multilingual text columns (`text`
and columns beginning with `text_`). See `DATA_LICENSE.md` for attribution and
licensing information. The bilingual Indonesian-English corpus is not used by
these source-group scripts and is not distributed here.

## Reproduction

Create an isolated Python environment, install the dependencies, and run the
commands from the repository root. Replace `<dataset.parquet>` with the path
to the prepared multilingual dataset.

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
python analysis/finetune_mbert_grouped.py --dataset <dataset.parquet> --output-dir results/finetuned_mbert_grouped --batch-size 32 --max-length 64 --max-epochs 1 --seeds 13 42 101
python analysis/summarize_finetuned_mbert.py --audit-dir results --bootstrap-repetitions 1000
```

The static-fusion experiments use one fixed source-group partition (split seed
42) and CNN seeds 13, 21, 42, 87, and 101. Download the official Qwen2.5
checkpoint to the path passed through `--model-path`; it is loaded in NF4 only
for offline vocabulary extraction. The PCA sweep excludes the padding entry
from fitted transformations and resets it to zero afterward. Conventional
baselines use five grouped split seeds. Fine-tuned mBERT uses seeds 13, 42,
and 101.

## Scope

This code reproduces the source-group experiments added during revision. The
historical message-level tables are retained in the manuscript only as explicitly
labeled exploratory results and are not the primary evidence of the revised
study.

## Citation

Until the manuscript receives its final bibliographic record, cite this
repository using `CITATION.cff` and DOI `10.5281/zenodo.22788881`.
