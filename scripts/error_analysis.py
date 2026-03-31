"""
Error analysis pipeline for robustness diagnostics.
"""

import argparse
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_OUTPUT = "reports/error_analysis.md"
DEFAULT_PRED_CSV = "reports/predictions_with_slices.csv"
DEFAULT_FIG_PATH = "reports/figures/error_distribution.png"


def detect_language(text):
    if not isinstance(text, str):
        return "english_only"

    hinglish_tokens = {
        "hai", "nahi", "bahut", "ka", "ki", "mein", "yahan", "wajah", "gadda", "gaadi", "jaldi", "ruk"
    }
    t = text.lower()

    if re.search(r"[^\x00-\x7F]", t):
        return "hinglish_mixed"

    tokens = re.findall(r"[a-zA-Z]+", t)
    return "hinglish_mixed" if any(tok in hinglish_tokens for tok in tokens) else "english_only"


def heuristic_predict(text):
    if not isinstance(text, str):
        return "congestion", 0.40

    t = text.lower()
    if any(k in t for k in ["accident", "collision", "crash", "slip"]):
        return "accident", 0.90
    if any(k in t for k in ["pothole", "gadda", "road damage"]):
        return "potholes", 0.86
    if any(k in t for k in ["traffic", "jam", "congestion", "slow"]):
        return "congestion", 0.82
    return "congestion", 0.52


def time_bucket(hour):
    try:
        h = int(hour)
    except Exception:
        return "unknown"

    if 6 <= h < 12:
        return "morning"
    if 12 <= h < 18:
        return "afternoon"
    if 18 <= h < 24:
        return "evening"
    return "night"


def load_predictions(test_csv):
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"test CSV not found: {test_csv}")

    df = pd.read_csv(test_csv)
    required = {"text", "label", "location", "hour", "weather"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "predicted_label" not in df.columns or "confidence" not in df.columns:
        if os.path.exists(DEFAULT_PRED_CSV):
            pred_df = pd.read_csv(DEFAULT_PRED_CSV)
            overlap_cols = [c for c in ["text", "predicted_label", "confidence"] if c in pred_df.columns]
            if set(["text", "predicted_label", "confidence"]).issubset(set(overlap_cols)):
                merged = df.merge(pred_df[["text", "predicted_label", "confidence"]], on="text", how="left")
                df["predicted_label"] = merged["predicted_label"]
                df["confidence"] = merged["confidence"]
                print(f"Merged predictions from {DEFAULT_PRED_CSV}")

    if "predicted_label" not in df.columns or df["predicted_label"].isna().all():
        preds = []
        confs = []
        for text in df["text"].fillna(""):
            p, c = heuristic_predict(text)
            preds.append(p)
            confs.append(c)
        df["predicted_label"] = preds
        df["confidence"] = confs
        print("Used heuristic predictions because no model predictions were available.")

    if "confidence" not in df.columns:
        df["confidence"] = 0.50

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.50)
    df["language_type"] = df["text"].apply(detect_language)
    df["time_bucket"] = df["hour"].apply(time_bucket)
    df["is_error"] = (df["predicted_label"].astype(str) != df["label"].astype(str)).astype(int)
    return df


def summarize_fp_fn(df):
    labels = sorted(df["label"].astype(str).unique().tolist())
    rows = []
    for label in labels:
        fp = int(((df["predicted_label"] == label) & (df["label"] != label)).sum())
        fn = int(((df["predicted_label"] != label) & (df["label"] == label)).sum())
        rows.append({"class": label, "false_positives": fp, "false_negatives": fn})
    return pd.DataFrame(rows)


def error_rate_per_class(df):
    rows = []
    for label, gdf in df.groupby("label"):
        total = len(gdf)
        errors = int((gdf["predicted_label"] != gdf["label"]).sum())
        rate = errors / total if total > 0 else 0.0
        rows.append({"class": label, "total": total, "errors": errors, "error_rate": rate})
    return pd.DataFrame(rows).sort_values("error_rate", ascending=False)


def save_error_plot(df, fig_path):
    err_df = df[df["is_error"] == 1].copy()
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)

    if err_df.empty:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "No errors to plot", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        return

    components = []
    for col in ["language_type", "weather", "time_bucket"]:
        counts = err_df[col].value_counts().rename_axis("bucket").reset_index(name="count")
        counts["group"] = col
        components.append(counts)

    plot_df = pd.concat(components, ignore_index=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="bucket", y="count", hue="group")
    plt.title("Error Distribution by Language, Weather, and Time")
    plt.xlabel("Bucket")
    plt.ylabel("Error Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def root_cause_text(df):
    err_df = df[df["is_error"] == 1].copy()
    if err_df.empty:
        return "No misclassifications found. Root cause analysis is not applicable."

    by_lang = err_df["language_type"].value_counts(normalize=True)
    by_weather = err_df["weather"].value_counts(normalize=True)
    by_time = err_df["time_bucket"].value_counts(normalize=True)

    top_lang = by_lang.index[0] if not by_lang.empty else "unknown"
    top_weather = by_weather.index[0] if not by_weather.empty else "unknown"
    top_time = by_time.index[0] if not by_time.empty else "unknown"

    share = by_lang.iloc[0] * 100 if not by_lang.empty else 0
    return (
        f"Primary error concentration appears in {top_lang} text during {top_time} under {top_weather} weather "
        f"(about {share:.1f}% of observed errors), likely due to lexical variability and OOV tokens."
    )


def build_markdown(df, output_path):
    err_df = df[df["is_error"] == 1].copy()
    top20 = err_df.sort_values("confidence", ascending=True).head(20)
    class_rates = error_rate_per_class(df)
    fp_fn = summarize_fp_fn(df)

    lines = []
    lines.append("# Error Analysis Report\n")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}\n")

    lines.append("## Top failure cases with explanations\n")
    if top20.empty:
        lines.append("No misclassified examples found in this run.\n")
    else:
        lines.append("| Text | Predicted | True | Confidence | Explanation |")
        lines.append("|---|---|---|---:|---|")
        for _, row in top20.iterrows():
            text = str(row["text"]).replace("|", " ").strip()
            text_short = (text[:120] + "...") if len(text) > 120 else text
            explanation = "Low-confidence mismatch; inspect tokenization and context coverage."
            lines.append(
                f"| {text_short} | {row['predicted_label']} | {row['label']} | {row['confidence']:.2f} | {explanation} |"
            )

    lines.append("\n## Error patterns by location, time, language, weather\n")
    for col in ["location", "time_bucket", "language_type", "weather"]:
        lines.append(f"### {col}\n")
        grp = (
            df.groupby(col)["is_error"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "errors", "count": "total"})
            .reset_index()
        )
        grp["error_rate"] = np.where(grp["total"] > 0, grp["errors"] / grp["total"], 0.0)

        lines.append("| Bucket | Errors | Total | Error Rate |")
        lines.append("|---|---:|---:|---:|")
        for _, row in grp.sort_values("error_rate", ascending=False).iterrows():
            lines.append(f"| {row[col]} | {int(row['errors'])} | {int(row['total'])} | {row['error_rate']:.2f} |")

    lines.append("\n## Error rate per class\n")
    lines.append("| Class | Total | Errors | Error Rate |")
    lines.append("|---|---:|---:|---:|")
    for _, row in class_rates.iterrows():
        lines.append(f"| {row['class']} | {int(row['total'])} | {int(row['errors'])} | {row['error_rate']:.2f} |")

    lines.append("\n## False positives vs false negatives\n")
    lines.append("| Class | False Positives | False Negatives |")
    lines.append("|---|---:|---:|")
    for _, row in fp_fn.iterrows():
        lines.append(f"| {row['class']} | {int(row['false_positives'])} | {int(row['false_negatives'])} |")

    lines.append("\n## Root cause analysis\n")
    lines.append(f"- {root_cause_text(df)}")
    lines.append(f"- Error distribution plot: {DEFAULT_FIG_PATH}")

    lines.append("\n## Recommendations for improvement\n")
    lines.append("- Increase Hinglish and weather-specific training samples.")
    lines.append("- Add confusion-aware hard-negative mining for commonly swapped classes.")
    lines.append("- Calibrate confidence threshold by slice rather than using a global threshold.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote error analysis report to {output_path}")


def main(test_csv, output):
    df = load_predictions(test_csv)
    save_error_plot(df, DEFAULT_FIG_PATH)
    build_markdown(df, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run error analysis for prediction failures")
    parser.add_argument("--test_csv", required=True, help="Path to test CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to markdown report")
    args = parser.parse_args()

    main(args.test_csv, args.output)
