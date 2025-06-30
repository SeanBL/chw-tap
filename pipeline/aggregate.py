import pandas as pd
from typing import List, Dict
from collections import Counter
import numpy as np
import os

def aggregate_concept_frequencies(ratings: List[Dict], model_names: List[str]) -> pd.DataFrame:
    """
    Returns a DataFrame with counts and mean scores per label per model.
    """
    rows = []
    for testimonial in ratings:
        for label, model_scores in testimonial["labels"].items():
            for model in model_names:
                score = model_scores.get(model, 0.0)
                rows.append({"label": label, "model": model, "score": score})

    df = pd.DataFrame(rows)
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