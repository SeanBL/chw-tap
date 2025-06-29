import pandas as pd
from typing import List, Dict
import os


def compute_model_disagreements(ratings: List[Dict[str, Dict[str, float]]], threshold: float = 0.5) -> pd.DataFrame:
    """
    Calculate binary disagreements per testimonial and label between models.
    Returns a DataFrame of disagreement records.
    """
    disagreements = []

    for testimonial in ratings:
        text = testimonial['text']
        for label, model_scores in testimonial['labels'].items():
            binary = {model: int(score >= threshold) for model, score in model_scores.items()}
            values = list(binary.values())
            if len(set(values)) > 1:
                disagreements.append({
                    "testimonial": text,
                    "label": label,
                    **binary
                })

    return pd.DataFrame(disagreements)


def summarize_disagreements(disagreement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize number of disagreements per label and per model.
    """
    if disagreement_df.empty:
        return pd.DataFrame()

    model_cols = [col for col in disagreement_df.columns if col not in ["testimonial", "label"]]
    summary_rows = []

    for label in disagreement_df['label'].unique():
        label_df = disagreement_df[disagreement_df['label'] == label]
        for model in model_cols:
            disagreement_count = sum(
                row[model] != max(set(list(row[model_cols])), key=list(row[model_cols]).count)
                for _, row in label_df.iterrows()
            )
            disagreement_pct = disagreement_count / len(label_df)
            summary_rows.append({
                "label": label,
                "model": model,
                "disagreements": disagreement_count,
                "total": len(label_df),
                "disagreement_pct": round(disagreement_pct, 2)
            })

    return pd.DataFrame(summary_rows)


def flag_high_disagreement_testimonials(disagreement_df: pd.DataFrame, model_names: List[str], threshold: int = 2) -> pd.DataFrame:
    """
    Identify testimonials with disagreements from at least `threshold` different models for the same label.
    """
    if disagreement_df.empty:
        return pd.DataFrame()

    flagged_rows = []

    grouped = disagreement_df.groupby(['testimonial', 'label'])
    for (text, label), group in grouped:
        disagreement_counts = []
        for _, row in group.iterrows():
            values = [row[model] for model in model_names if model in row]
            if len(set(values)) > 1:
                disagreement_counts.append(1)
        if sum(disagreement_counts) >= threshold:
            flagged_rows.append({
                "testimonial": text,
                "label": label,
                "disagreement_instances": sum(disagreement_counts)
            })

    return pd.DataFrame(flagged_rows)


def export_disagreements_to_excel(
        disagreement_df: pd.DataFrame, 
        summary_df: pd.DataFrame, 
        flagged_df: pd.DataFrame,
        model_pct_df: pd.DataFrame, 
        out_path: str
):
    """
    Save disagreement log, summary statistics, and flagged rows to Excel.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        disagreement_df.to_excel(writer, sheet_name="Disagreements", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if not flagged_df.empty:
            flagged_df.to_excel(writer, sheet_name="Flagged", index=False)
        if not model_pct_df.empty:
            model_pct_df.to_excel(writer, sheet_name="Model Contributions", index=False)

def model_disagreement_percentages(disagreement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame showing what percent of disagreements each model contributes to, per label.
    """
    if disagreement_df.empty:
        return pd.DataFrame()

    model_cols = [col for col in disagreement_df.columns if col not in ["testimonial", "label"]]
    total_counts = disagreement_df.groupby("label").size().to_dict()

    results = []
    for label in disagreement_df["label"].unique():
        label_df = disagreement_df[disagreement_df["label"] == label]
        for model in model_cols:
            # Count how often the model disagrees with majority vote
            disagreements = label_df.apply(
                lambda row: row[model] != max(set([row[m] for m in model_cols]), key=[row[m] for m in model_cols].count),
                axis=1
            ).sum()
            pct = disagreements / total_counts[label]
            results.append({
                "label": label,
                "model": model,
                "disagreement_count": disagreements,
                "total_disagreements": total_counts[label],
                "percent_of_label_disagreements": round(pct * 100, 2)
            })

    return pd.DataFrame(results)

