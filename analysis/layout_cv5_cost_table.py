"""Replace the static-classifier cost table with matched five-fold measurements."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "output/revised_manuscript_draft/sn-article.tex"
RUNS = ROOT / "output/source_group_cv5/all_test_fold_runs.csv"
SPARSE_EFFICIENCY = (ROOT / "output/source_group_cv5/classical"
                     / "grouped_baseline_efficiency.csv")
NAMES = (
    ("mBERT", "mBERT"),
    ("mBERT PCA-256", "mBERT PCA-256"),
    ("mBERT PCA-512", "mBERT PCA-512"),
    ("Qwen2.5 raw", "Qwen2.5 raw"),
    ("Qwen2.5+PCA-1024", "Qwen2.5 PCA-1024"),
    ("mBERT+Qwen concat no PCA", "mBERT+Qwen2.5 concat"),
    ("mBERT+Qwen RP-1024", "mBERT+Qwen2.5 RP-1024"),
    ("mBERT+Qwen PCA-1024", "mBERT+Qwen2.5 PCA-1024"),
    ("mBERT+Qwen+Char concat no PCA", "Three-source concat"),
    ("mBERT+Qwen+Char PCA-1024", "Three-source PCA-1024"),
)


def mean_sd(values: pd.Series, decimals: int) -> str:
    return f"${values.mean():,.{decimals}f}\\pm{values.std(ddof=1):,.{decimals}f}$"


def summarized_pair(row: pd.Series, column: str, decimals: int) -> str:
    return (f"${row[f'{column}_mean']:,.{decimals}f}"
            f"\\pm{row[f'{column}_std']:,.{decimals}f}$")


def replace_table(source: str, label: str, replacement: str) -> str:
    label_at = source.index("\\label{" + label + "}")
    start = source.rfind("\\begin{table*}[t]", 0, label_at)
    end = source.index("\\end{table*}", label_at) + len("\\end{table*}")
    return source[:start] + replacement + source[end:]


def main() -> None:
    runs = pd.read_csv(RUNS)
    runs = runs[runs["family"] == "static embedding CNN"]
    lines = [
        "\\begin{table*}[t]\n\\centering\n",
        "\\caption{Downstream cost of the static-embedding CNN configurations "
        "across five source-group folds. Means and standard deviations are "
        "computed over the five training vocabularies and classifier runs; "
        "offline pretrained-encoder extraction and PCA fitting are excluded "
        "from CNN training and inference times.}\n",
        "\\label{tab:qwen-grouped-cost}\n\\label{tab:new-fusion-cost}\n",
        "\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n",
        "\\begin{tabular}{lccccc}\n\\toprule\n",
        "Configuration & Frozen table (MB) & Trainable parameters & Training (s) & "
        "Inference (ms/message) & Peak GPU (MB) \\\\\n",
        "\\midrule\n",
    ]
    for name, label in NAMES:
        rows = runs[runs["configuration"] == name]
        if len(rows) != 5:
            raise ValueError(f"Expected five folds for {name}, found {len(rows)}")
        parameters = rows["trainable_parameters"].unique()
        if len(parameters) != 1:
            raise ValueError(f"Parameter count varies for fixed-width model {name}")
        cells = [
            label,
            mean_sd(rows["stored_embedding_mb"], 2),
            f"{int(parameters[0]):,}",
            mean_sd(rows["train_seconds"], 2),
            mean_sd(rows["inference_ms_per_message"], 4),
            mean_sd(rows["peak_gpu_memory_mb"], 1),
        ]
        lines.append(" & ".join(cells) + " \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n}\n\\end{table*}")
    text = TEX.read_text(encoding="utf-8")
    text = replace_table(text, "tab:qwen-grouped-cost", "".join(lines))

    sparse = pd.read_csv(SPARSE_EFFICIENCY).set_index("model")
    sparse_names = (
        ("Char-TFIDF + LR", "Character TF--IDF + LR"),
        ("Char-TFIDF + Linear SVM", "Character TF--IDF + linear SVM"),
        ("Word-TFIDF + LR", "Word TF--IDF + LR"),
        ("Word-TFIDF + Linear SVM", "Word TF--IDF + linear SVM"),
        ("Word-TFIDF + SVD + HistGB", "Word TF--IDF + SVD-128 + HistGB"),
        ("Word-TFIDF + SVD + MLP", "Word TF--IDF + SVD-128 + MLP"),
    )
    lines = [
        "\\begin{table*}[t]\n\\centering\n",
        "\\caption{Computational cost of document-level sparse baselines across "
        "the five source-group folds (mean $\\pm$ standard deviation). "
        "Feature fitting and classifier fitting are reported separately; "
        "inference includes text vectorization and prediction.}\n",
        "\\label{tab:grouped-efficiency}\n",
        "\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n",
        "\\begin{tabular}{lcccc}\n\\toprule\n",
        "Model & Feature fit (s) & Classifier fit (s) & "
        "Inference (ms/message) & Serialized pipeline (MB) \\\\\n",
        "\\midrule\n",
    ]
    for name, label in sparse_names:
        row = sparse.loc[name]
        cells = [
            label,
            summarized_pair(row, "vectorizer_fit_seconds", 2),
            summarized_pair(row, "classifier_fit_seconds", 2),
            summarized_pair(row, "inference_ms_per_message", 3),
            summarized_pair(row, "serialized_model_mb", 2),
        ]
        lines.append(" & ".join(cells) + " \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n}\n\\end{table*}")
    text = replace_table(text, "tab:grouped-efficiency", "".join(lines))
    TEX.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
