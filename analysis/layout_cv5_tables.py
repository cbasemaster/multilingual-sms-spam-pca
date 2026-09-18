"""Replace grouped-result tables with matched five-fold test summaries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "output/revised_manuscript_draft/sn-article.tex"
SUMMARY = ROOT / "output/source_group_cv5/all_test_fold_summary.csv"
RUNS = ROOT / "output/source_group_cv5/all_test_fold_runs.csv"
METRICS = ("accuracy", "spam_f1", "macro_f1", "mcc", "cohen_kappa",
           "roc_auc", "ece_15")


def family(name: str) -> str:
    if name == "Fine-tuned mBERT":
        return "Contextual baseline"
    if name.startswith("mBERT+Qwen+Char"):
        return "mBERT + Qwen2.5 + character TF--IDF"
    if name.startswith("mBERT+Qwen"):
        return "mBERT + Qwen2.5"
    if name.startswith("DistilBERT+mBERT") or name == "DistilBERT":
        return "DistilBERT + mBERT"
    if name.startswith("Qwen2.5"):
        return "Qwen2.5"
    if name.startswith("Char-TFIDF"):
        return "Character TF--IDF"
    if name.startswith("Word-TFIDF"):
        return "Word TF--IDF"
    if name == "mBERT" or name.startswith("mBERT PCA-"):
        return "mBERT"
    raise ValueError(f"Unclassified CV configuration: {name}")


def display(name: str) -> str:
    if name == "Fine-tuned mBERT":
        return name
    for prefix in ("mBERT+Qwen+Char ", "mBERT+Qwen ",
                   "DistilBERT+mBERT "):
        if name.startswith(prefix):
            method = name[len(prefix):]
            method = method.replace("concat no PCA", "Concat (no PCA)")
            method = method.replace("standardized concat", "Standardized concat")
            method = method.replace("raw concat", "Raw concat")
            return method + " + CNN"
    if name.startswith("Qwen2.5"):
        return name.replace("Qwen2.5+", "").replace("Qwen2.5 raw", "No PCA") + " + CNN"
    if name in ("mBERT", "DistilBERT"):
        return ("No PCA" if name == "mBERT" else "DistilBERT only") + " + CNN"
    if name.startswith("mBERT PCA-"):
        return name.removeprefix("mBERT ") + " + CNN"
    for prefix in ("Char-TFIDF + ", "Word-TFIDF + "):
        if name.startswith(prefix):
            method = name[len(prefix):].replace("Linear SVM", "linear SVM")
            if method in ("LR", "linear SVM"):
                method = "No PCA + " + method
            return method
    raise ValueError(f"Unrecognized display name: {name}")


def integer_tex(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def dimensions(name: str, records: pd.DataFrame) -> tuple[str, int | None]:
    if name == "Fine-tuned mBERT":
        return "768 hidden", None
    if "validation-selected" in name:
        values = sorted(set(records["final_dimension"].dropna().astype(int)))
        if len(values) == 1:
            return f"$4{{,}}352\\rightarrow{integer_tex(values[0])}$", values[0]
        return "fold-specific", None
    row = records.iloc[0]
    final = row.get("final_dimension")
    initial = row.get("input_dimension")
    if pd.isna(final):
        final = row.get("feature_dimension")
    if pd.isna(initial):
        initial = 75_000 if "SVD" in name else final
    initial, final = int(initial), int(final)
    if initial != final:
        return f"${integer_tex(initial)}\\rightarrow{integer_tex(final)}$", final
    return integer_tex(final), final


def decorate(value: str, command: str) -> str:
    return "$\\" + command + "{" + value[1:-1] + "}$"


def row_order(row: dict) -> tuple[int, int, str]:
    name = row["name"]
    if "validation-selected" in name:
        return (0, -1, name)
    if "PCA" in name and "no PCA" not in name:
        return (1, row["width"] or 99999, name)
    if "RP" in name or "SVD" in name:
        return (3, row["width"] or 99999, name)
    return (2, row["width"] or 99999, name)


def make_table(caption: str, labels: list[str], families: list[str],
               rows: dict[str, list[dict]], note: str) -> str:
    out = [
        "\\begin{table*}[t]\n\\centering\n",
        "\\caption{" + caption + "}\n",
        *("\\label{" + label + "}\n" for label in labels),
        "\\scriptsize\n\\setlength{\\tabcolsep}{1.5pt}\n",
        "\\renewcommand{\\arraystretch}{1.02}\n",
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llccccccc}\n",
        "\\toprule\n",
        "Configuration & Dimensions & Acc. & Spam F1 & Macro F1 & MCC & $\\kappa$ & AUC & ECE \\\\\n",
        "\\midrule\n",
    ]
    for group_index, heading in enumerate(families):
        group = sorted(rows[heading], key=row_order)
        for metric in METRICS:
            ordered = sorted((row for row in group if pd.notna(row["means"][metric])),
                             key=lambda row: (
                row["means"][metric] if metric == "ece_15" else -row["means"][metric],
                row["sds"][metric],
                -row["means"]["mcc"], row["name"]
            ))
            distinct = []
            for row in ordered:
                displayed_mean = round(row["means"][metric], 4)
                if displayed_mean not in distinct:
                    distinct.append(displayed_mean)
            for rank, mean in enumerate(distinct[:2]):
                winner = next(row for row in ordered
                              if round(row["means"][metric], 4) == mean)
                winner["metric_style"][metric] = "mathbf" if rank == 0 else "underline"
        widths = sorted({row["width"] for row in group if row["width"] is not None})
        if len(widths) > 1:
            for rank, width in enumerate(widths[:2]):
                choices = [row for row in group if row["width"] == width]
                winner = sorted(choices, key=lambda row: (
                    -row["means"]["mcc"], row["sds"]["mcc"], row["name"]
                ))[0]
                winner["dimension_style"] = "mathbf" if rank == 0 else "underline"
        out.append("\\multicolumn{9}{l}{\\textit{" + heading + "}} \\\\\n")
        for row in group:
            dim = row["dimension"]
            if row["dimension_style"]:
                command = row["dimension_style"]
                if dim.startswith("$") and "\\rightarrow" in dim:
                    before, after = dim[1:-1].rsplit("\\rightarrow", 1)
                    dim = "$" + before + "\\rightarrow\\" + command + "{" + after + "}$"
                else:
                    dim = "\\textbf{" + dim + "}" if command == "mathbf" else "\\underline{" + dim + "}"
            cells = [display(row["name"]), dim]
            for metric in METRICS:
                if pd.isna(row["means"][metric]):
                    cells.append("--")
                    continue
                value = f"${row['means'][metric]:.4f}\\pm{row['sds'][metric]:.4f}$"
                if metric in row["metric_style"]:
                    value = decorate(value, row["metric_style"][metric])
                cells.append(value)
            out.append(" & ".join(cells) + " \\\\\n")
        if group_index != len(families) - 1:
            out.append("\\midrule\n")
    out.extend([
        "\\bottomrule\n\\end{tabular}\n}\n",
        "\\vspace{2pt}\n\\begin{minipage}{\\textwidth}\n\\scriptsize\n",
        "\\textit{Note.} " + note + "\n",
        "\\end{minipage}\n\\end{table*}",
    ])
    return "".join(out)


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    runs = pd.read_csv(RUNS)
    grouped = defaultdict(list)
    for _, item in summary.iterrows():
        name = item["configuration"]
        if name == "Fine-tuned mBERT":
            continue
        source = runs[(runs["family"] == item["family"])
                      & (runs["configuration"] == name)]
        if len(source) != 5:
            raise ValueError(f"Incomplete five-fold result: {name}")
        dim, width = dimensions(name, source)
        grouped[family(name)].append({
            "name": name, "dimension": dim, "width": width,
            "means": {metric: float(item[f"{metric}_mean"]) for metric in METRICS},
            "sds": {metric: float(item[f"{metric}_std"]) for metric in METRICS},
            "metric_style": {}, "dimension_style": None,
        })
    families = ["mBERT + Qwen2.5", "mBERT + Qwen2.5 + character TF--IDF",
                "DistilBERT + mBERT", "Qwen2.5", "mBERT",
                "Character TF--IDF", "Word TF--IDF"]
    if any(not grouped[name] for name in families):
        raise ValueError("Missing a feature family")
    for heading in families:
        names = [row["name"] for row in grouped[heading]]
        has_pca = any("PCA-" in name for name in names)
        has_no_pca = any(
            "PCA" not in name or "no PCA" in name for name in names
        )
        if not (has_pca and has_no_pca):
            raise ValueError(f"Missing PCA/no-PCA pair in {heading}")
    rank_note = (
        "Bold denotes the best mean and underline the second-best within each "
        "feature family. Equal means have one marked representative, chosen by "
        "lower SD then higher MCC; lower ECE is better. A lower final width is "
        "preferable; equal widths "
        "have one representative. All metrics are held-out test results from five "
        "disjoint source-group folds, with one fixed training seed per fold. "
        "Every displayed feature family includes PCA and no-PCA configurations. "
        "The fine-tuned contextual baseline is reported separately in Section 5.4. "
        "AUC and ECE were not recorded for sparse controls."
    )
    table = make_table(
        "PCA-based embedding fusion and matched sparse controls under five "
        "source-group test folds (mean $\\pm$ standard deviation).",
        ["tab:grouped-baselines", "tab:qwen-grouped-results",
         "tab:new-fusion-results", "tab:five-split-sparse-baselines"],
        families, grouped, rank_note,
    )
    source = TEX.read_text(encoding="utf-8")
    label = source.index("\\label{tab:grouped-baselines}")
    start = source.rfind("\\begin{table*}[t]", 0, label)
    second = source.index("\\label{tab:five-split-sparse-baselines}", label)
    end = source.index("\\end{table*}", second) + len("\\end{table*}")
    TEX.write_text(source[:start] + table + source[end:], encoding="utf-8")
    print({heading: len(grouped[heading]) for heading in families})


if __name__ == "__main__":
    main()
