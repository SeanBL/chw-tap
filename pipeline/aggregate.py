import pandas as pd
from typing import List, Dict
from collections import Counter
import numpy as np
import os

def aggregate_concept_frequencies(ratings: List[Dict], model_names: List[str]) -> pd.DataFrame:
    """
    Returns a DataFrame with counts and mean scores per label per model.
    Assumes each testimonial entry corresponds to a single model.
    """
    rows = []
    print("DEBUG: First rating entry:", ratings[0] if ratings else "No data")
    for testimonial in ratings:
        model = testimonial.get("model", "unknown")  # fallback to "unknown" if missing
        labels = testimonial.get("labels", {})
        if not isinstance(labels, dict):
            continue  # skip malformed rows

        for label, model_scores in labels.items():
            if isinstance(model_scores, dict):
                for model, score in model_scores.items():
                    rows.append({
                        "label": label,
                        "model": model,
                        "score": score
                    })
            else:
                print(f"⚠️ Unexpected format for label '{label}':", model_scores)

    df = pd.DataFrame(rows)
    print("DEBUG: df.columns =", df.columns.tolist())
    print(df.head())

    if df.empty:
        print("⚠️ No label scores found in the ratings. Skipping frequency aggregation.")
        return pd.DataFrame(columns=["label", "model", "count", "mean_score"])

    return df.groupby(["label", "model"]).agg(
        count=("score", "count"),
        mean_score=("score", "mean")
    ).reset_index()

def compute_consensus_labels(ratings: List[Dict], model_names: List[str], method: str = "vote", threshold: float = 0.5) -> List[Dict]:
    """
    Compute consensus labels per testimonial using vote or mean aggregation.
    Returns a list of consensus label dictionaries per testimonial.
    """
    consensus_results = []

    for testimonial in ratings:
        consensus = {}
        for label, model_scores in testimonial["labels"].items():
            scores = [model_scores.get(model, 0.0) for model in model_names]
            if method == "vote":
                binary = [int(score >= threshold) for score in scores]
                consensus[label] = int(sum(binary) >= (len(binary) / 2))
            elif method == "mean":
                consensus[label] = np.mean(scores)
            else:
                raise ValueError(f"Unknown consensus method: {method}")
        consensus_results.append({
            "text": testimonial["text"],
            "consensus_labels": consensus
        })

    return consensus_results

def export_consensus_to_excel(consensus_data: List[Dict], out_path: str):
    """
    Export consensus labels to an Excel file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(consensus_data)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Consensus Labels", index=False)