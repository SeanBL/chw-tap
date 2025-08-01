import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
from pingouin import intraclass_corr
import krippendorff
from typing import List, Dict


def compute_irr_scores(ratings: List[Dict[str, Dict[str, float]]], threshold: float = 0.5) -> Dict:
    """
    Computes IRR scores across multiple models for each label and overall.
    Input:
        ratings: List of testimonials with per-label ratings per model.
        threshold: Cutoff for converting scores to binary for Fleiss/Cohen/etc.
    Output:
        Dictionary with ICC, Fleiss, Cohen, Krippendorff, and % Agreement.
    """
    all_labels = list(ratings[0]["labels"].keys())
    model_names = list(next(iter(ratings[0]["labels"].values())).keys())

    per_label_results = {}
    all_scores_matrix = []  # For overall Krippendorff

    for label in all_labels:
        label_scores = []  # Continuous for ICC, Krippendorff
        binary_scores = []  # Binarized for Fleiss, Cohen, % Agreement

        for testimonial in ratings:
            model_scores = testimonial["labels"][label]
            row = [model_scores.get(model, 0.0) for model in model_names]
            label_scores.append(row)
            binary_scores.append([int(score >= threshold) for score in row])
            all_scores_matrix.append(row)  # Flattened for Krippendorff overall

        df = pd.DataFrame(label_scores, columns=model_names)
        df_long = pd.melt(df.reset_index(), id_vars=['index'], var_name='rater', value_name='score')

        icc = intraclass_corr(data=df_long, targets='index', raters='rater', ratings='score')['ICC'].mean()

        # Convert binary scores into contingency table
        fleiss_input = []
        for row in binary_scores:
            counts = [row.count(0), row.count(1)]
            fleiss_input.append(counts)

        fleiss = fleiss_kappa(np.array(fleiss_input))

        cohen_scores = []
        cohen_notes = None
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                a = [row[i] for row in binary_scores]
                b = [row[j] for row in binary_scores]
                if len(set(a)) == 1 and len(set(b)) == 1 and set(a) == set(b):
                    cohen_notes = f"No variation in binary labels for models {model_names[i]} vs {model_names[j]}"
                    continue  # skip this pair
                try:
                    score = cohen_kappa_score(a, b)
                    if not np.isnan(score):
                        cohen_scores.append(score)
                except Exception as e:
                    print(f"[WARN] Cohen Kappa failed for models {model_names[i]} vs {model_names[j]}: {e}")

        cohen = round(np.mean(cohen_scores), 3) if cohen_scores else "N/A"

        # Krippendorff's alpha with variability check

        flat_values = list(np.array(label_scores).flatten())
        unique_values = set(flat_values)

        if len(unique_values) < 2:
            kripp = "N/A (no variability)"
            print(f"[INFO] Skipping Krippendorff for label '{label}' due to identical scores: {unique_values}")
        else:
            try:
                kripp = krippendorff.alpha(
                    reliability_data=np.array(label_scores).T,
                    level_of_measurement='interval'
                )
                kripp = round(kripp, 3)
            except Exception as e:
                kripp = "N/A"
                print(f"[WARN] Krippendorff failed for label '{label}': {e}")

        percent = np.mean([
            len(set([row[i] >= threshold for i in range(len(row))])) == 1
            for row in binary_scores
        ])

        per_label_results[label] = {
            "icc": round(icc, 3),
            "fleiss": round(fleiss, 3),
            "cohen": cohen,
            "krippendorff": kripp,
            "percent_agreement": round(percent, 3)
        }

        if cohen_notes:
            per_label_results[label]["cohen_notes"] = cohen_notes

    # Overall Krippendorff
    flat_all_values = list(np.array(all_scores_matrix).flatten())
    if len(set(flat_all_values)) < 2:
        overall_kripp = "N/A (no variability)"
        print("[INFO] Skipping overall Krippendorff: all values are identical.")
    else:
        try:
            overall_kripp = krippendorff.alpha(
                reliability_data=np.array(all_scores_matrix).T,
                level_of_measurement='interval'
            )
            overall_kripp = round(overall_kripp, 3)
        except Exception as e:
            overall_kripp = "N/A"
            print(f"[WARN] Krippendorff failed for overall: {e}")

    return {
        "per_label": per_label_results,
        "overall": {
            "krippendorff": overall_kripp,
        }
    }

