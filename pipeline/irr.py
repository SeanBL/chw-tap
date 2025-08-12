import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
from pingouin import intraclass_corr
import krippendorff
from typing import List, Dict

def _to_float_or_nan(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def _round_or_nan(x, ndigits=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return round(float(x), ndigits)
    except Exception:
        return np.nan

def _binarize_row(row, threshold):
    """Convert a list of floats (may include NaN) to 0/1 with NaN->0 default."""
    row = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in row]
    return [int(v >= threshold) for v in row]

def compute_irr_scores(ratings: List[Dict[str, Dict[str, float]]], threshold: float = 0.5) -> Dict:
    """
    Computes IRR scores across multiple models for each label and overall.
    Numeric fields are always floats (or np.nan). Explanatory strings go in *_notes.
    """
    if not ratings:
        return {"per_label": {}, "overall": {"krippendorff": np.nan}}

    # Collect labels and model names robustly
    all_labels = list(ratings[0]["labels"].keys())
    # Union of model keys across the first item's labels (covers missing providers)
    first_labels_block = ratings[0]["labels"]
    model_names = sorted({m for lbl in first_labels_block for m in first_labels_block[lbl].keys()})

    per_label_results = {}
    all_scores_matrix = []  # For overall Krippendorff (continuous)

    for label in all_labels:
        label_scores = []   # continuous, shape: (n_items, n_models)
        binary_scores = []  # binarized

        for testimonial in ratings:
            model_scores = testimonial["labels"][label]
            row = [_to_float_or_nan(model_scores.get(model)) for model in model_names]
            label_scores.append(row)
            binary_scores.append(_binarize_row(row, threshold))
            all_scores_matrix.append(row)

        # Build DataFrame for ICC
        df = pd.DataFrame(label_scores, columns=model_names)
        df_long = pd.melt(
            df.reset_index(),
            id_vars=["index"],
            var_name="rater",
            value_name="score",
        )

        # ICC (handle failures/degenerate cases)
        icc_val = np.nan
        try:
            icc_tbl = intraclass_corr(
                data=df_long, targets="index", raters="rater", ratings="score"
            )
            # Use mean of ICC column; pingouin returns multiple ICC forms
            icc_val = pd.to_numeric(icc_tbl["ICC"], errors="coerce").mean()
        except Exception as e:
            # leave as NaN
            pass

        # Fleiss' kappa (binary categories 0/1)
        fleiss_val = np.nan
        try:
            fleiss_input = []
            for row in binary_scores:
                # counts of 0 and 1 across raters
                counts = [row.count(0), row.count(1)]
                fleiss_input.append(counts)
            fleiss_val = fleiss_kappa(np.array(fleiss_input))
        except Exception:
            pass

        # Pairwise Cohen's kappa on binarized data
        cohen_scores = []
        cohen_notes = None
        try:
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    a = [row[i] for row in binary_scores]
                    b = [row[j] for row in binary_scores]
                    # If both are constant and identical, skip (undefined kappa)
                    if len(set(a)) == 1 and set(a) == set(b):
                        cohen_notes = (
                            f"No variation in binary labels for models "
                            f"{model_names[i]} vs {model_names[j]}"
                        )
                        continue
                    k = cohen_kappa_score(a, b)
                    if k is not None and not np.isnan(k):
                        cohen_scores.append(float(k))
        except Exception:
            pass

        cohen_val = np.nan if not cohen_scores else float(np.mean(cohen_scores))

        # Krippendorff (interval) supports missing values as NaN; shape: (n_raters, n_items)
        kripp_val = np.nan
        try:
            arr = np.array(label_scores, dtype=float)
            # If everything identical (or all NaN), kripp is undefined -> NaN
            finite_vals = arr[np.isfinite(arr)]
            if finite_vals.size >= 2 and np.nanstd(finite_vals) > 0:
                kripp_val = krippendorff.alpha(
                    reliability_data=arr.T,  # (raters, items)
                    level_of_measurement="interval",
                )
        except Exception:
            pass

        # Percent agreement on binary decisions (treat NaN as 0)
        percent = np.nan
        try:
            agree_flags = [
                int(len(set(row)) == 1) for row in binary_scores  # all raters same for this item?
            ]
            if agree_flags:
                percent = float(np.mean(agree_flags))
        except Exception:
            pass

        # Store rounded numeric values; keep notes as strings
        entry = {
            "icc": _round_or_nan(icc_val),
            "fleiss": _round_or_nan(fleiss_val),
            "cohen": _round_or_nan(cohen_val),
            "krippendorff": _round_or_nan(kripp_val),
            "percent_agreement": _round_or_nan(percent),
        }
        if cohen_notes:
            entry["cohen_notes"] = cohen_notes
        per_label_results[label] = entry

    # Overall Krippendorff over all labels/items (interval)
    overall_kripp = np.nan
    try:
        arr_all = np.array(all_scores_matrix, dtype=float)
        finite_vals = arr_all[np.isfinite(arr_all)]
        if finite_vals.size >= 2 and np.nanstd(finite_vals) > 0:
            overall_kripp = krippendorff.alpha(
                reliability_data=arr_all.T, level_of_measurement="interval"
            )
    except Exception:
        pass

    return {
        "per_label": per_label_results,
        "overall": {"krippendorff": _round_or_nan(overall_kripp)},
    }



# def compute_irr_scores(ratings: List[Dict[str, Dict[str, float]]], threshold: float = 0.5) -> Dict:
#     """
#     Computes IRR scores across multiple models for each label and overall.
#     Input:
#         ratings: List of testimonials with per-label ratings per model.
#         threshold: Cutoff for converting scores to binary for Fleiss/Cohen/etc.
#     Output:
#         Dictionary with ICC, Fleiss, Cohen, Krippendorff, and % Agreement.
#     """
#     all_labels = list(ratings[0]["labels"].keys())
#     model_names = list(next(iter(ratings[0]["labels"].values())).keys())

#     per_label_results = {}
#     all_scores_matrix = []  # For overall Krippendorff

#     for label in all_labels:
#         label_scores = []  # Continuous for ICC, Krippendorff
#         binary_scores = []  # Binarized for Fleiss, Cohen, % Agreement

#         for testimonial in ratings:
#             model_scores = testimonial["labels"][label]
#             row = [model_scores.get(model, 0.0) for model in model_names]
#             label_scores.append(row)
#             binary_scores.append([int(score >= threshold) for score in row])
#             all_scores_matrix.append(row)  # Flattened for Krippendorff overall

#         df = pd.DataFrame(label_scores, columns=model_names)
#         df_long = pd.melt(df.reset_index(), id_vars=['index'], var_name='rater', value_name='score')

#         icc = intraclass_corr(data=df_long, targets='index', raters='rater', ratings='score')['ICC'].mean()

#         # Convert binary scores into contingency table
#         fleiss_input = []
#         for row in binary_scores:
#             counts = [row.count(0), row.count(1)]
#             fleiss_input.append(counts)

#         fleiss = fleiss_kappa(np.array(fleiss_input))

#         cohen_scores = []
#         cohen_notes = None
#         for i in range(len(model_names)):
#             for j in range(i + 1, len(model_names)):
#                 a = [row[i] for row in binary_scores]
#                 b = [row[j] for row in binary_scores]
#                 if len(set(a)) == 1 and len(set(b)) == 1 and set(a) == set(b):
#                     cohen_notes = f"No variation in binary labels for models {model_names[i]} vs {model_names[j]}"
#                     continue  # skip this pair
#                 try:
#                     score = cohen_kappa_score(a, b)
#                     if not np.isnan(score):
#                         cohen_scores.append(score)
#                 except Exception as e:
#                     print(f"[WARN] Cohen Kappa failed for models {model_names[i]} vs {model_names[j]}: {e}")

#         cohen = round(np.mean(cohen_scores), 3) if cohen_scores else "N/A"

#         # Krippendorff's alpha with variability check

#         flat_values = list(np.array(label_scores).flatten())
#         unique_values = set(flat_values)

#         if len(unique_values) < 2:
#             kripp = "N/A (no variability)"
#             print(f"[INFO] Skipping Krippendorff for label '{label}' due to identical scores: {unique_values}")
#         else:
#             try:
#                 kripp = krippendorff.alpha(
#                     reliability_data=np.array(label_scores).T,
#                     level_of_measurement='interval'
#                 )
#                 kripp = round(kripp, 3)
#             except Exception as e:
#                 kripp = "N/A"
#                 print(f"[WARN] Krippendorff failed for label '{label}': {e}")

#         percent = np.mean([
#             len(set([row[i] >= threshold for i in range(len(row))])) == 1
#             for row in binary_scores
#         ])

#         per_label_results[label] = {
#             "icc": round(icc, 3),
#             "fleiss": round(fleiss, 3),
#             "cohen": cohen,
#             "krippendorff": kripp,
#             "percent_agreement": round(percent, 3)
#         }

#         if cohen_notes:
#             per_label_results[label]["cohen_notes"] = cohen_notes

#     # Overall Krippendorff
#     flat_all_values = list(np.array(all_scores_matrix).flatten())
#     if len(set(flat_all_values)) < 2:
#         overall_kripp = "N/A (no variability)"
#         print("[INFO] Skipping overall Krippendorff: all values are identical.")
#     else:
#         try:
#             overall_kripp = krippendorff.alpha(
#                 reliability_data=np.array(all_scores_matrix).T,
#                 level_of_measurement='interval'
#             )
#             overall_kripp = round(overall_kripp, 3)
#         except Exception as e:
#             overall_kripp = "N/A"
#             print(f"[WARN] Krippendorff failed for overall: {e}")

#     return {
#         "per_label": per_label_results,
#         "overall": {
#             "krippendorff": overall_kripp,
#         }
#     }

