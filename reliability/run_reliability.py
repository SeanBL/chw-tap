# run_reliability.py
import argparse, json, os, pathlib
import numpy as np
import pandas as pd

from reliability_metrics import (
    threshold_continuous_to_binary,
    bootstrap_alpha_nominal,
    permutation_test_alpha_nominal,
    icc_analytic_ci,
    icc_2_1_and_2_k,
    bootstrap_overall_alpha_from_concepts,
    alpha_interval_point,
    bootstrap_alpha_interval,
    bootstrap_overall_alpha_interval_from_concepts,
)

# ---------- Defaults so you don't need to pass these flags ----------
DEFAULT_THRESHOLD = 0.5          # for alpha binarization
DEFAULT_BOOTS = 5000             # bootstrap reps for alpha/interval-alpha CIs
DEFAULT_PERMS = 2000             # permutation reps for alpha>0
DEFAULT_ALPHA = 0.05             # for analytical ICC CI
DEFAULT_OUTDIR = "reliability_out"
DEFAULT_FORMAT = "both"          # "json", "excel", or "both"
DEFAULT_SHEET_INDEX = 0          # if Excel and no sheet provided

# ----------------------------
# Helpers (flags, prevalence, clipping)
# ----------------------------
def _clip_01(x):
    """Clip to [0,1] and zero out tiny negatives from numeric noise."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return x
    y = max(0.0, min(1.0, float(x)))
    if abs(y) < 1e-12:
        y = 0.0
    return y

def _alpha_flag(a, lo, hi):
    # Krippendorff guidance: >= .80 reliable; .667-.80 tentative
    if a is None or np.isnan(a):
        return "NA"
    if hi < 0.667:
        return "red"
    if lo >= 0.80:
        return "green"
    return "yellow"

def _icc_band(v):
    # Koo & Li (2016)
    if v is None or np.isnan(v):
        return "NA"
    if v < 0.50: return "poor"
    if v < 0.75: return "moderate"
    if v < 0.90: return "good"
    return "excellent"

def _prevalence(bin_mat: np.ndarray):
    valid = ~np.isnan(bin_mat)
    denom = valid.sum()
    if denom == 0:
        return np.nan
    pos = ((bin_mat == 1.0) & valid).sum()
    return float(pos / denom)

def _n_items_ge2(mat: np.ndarray):
    return int((~np.isnan(mat)).sum(axis=1 >= 0))  # placeholder (unused)

# ----------------------------
# Input loaders (JSON & Excel)
# ----------------------------

def load_json_any(path: str) -> list[dict]:
    ext = pathlib.Path(path).suffix.lower()
    data = []
    if ext in (".jsonl", ".jl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            data = obj
        elif isinstance(obj, dict):
            data = [obj]
        else:
            raise ValueError("JSON must be an object or a list of objects.")
    else:
        raise ValueError("Unsupported JSON extension. Use .json / .jsonl / .jl")
    return data

def json_to_long(df_like: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in df_like:
        tid = str(rec.get("id", "")).strip()
        labels = rec.get("labels", {}) or {}
        for concept, scores in labels.items():
            cname = str(concept).strip()
            if isinstance(scores, dict):
                for rater_key, val in scores.items():
                    if val is None:
                        continue
                    try:
                        score = float(val)
                    except Exception:
                        continue
                    rows.append({
                        "testimonial_id": tid,
                        "concept": cname,
                        "rater": str(rater_key).strip(),
                        "score": score
                    })
    if not rows:
        raise ValueError("No rows parsed from JSON. Check structure.")
    return pd.DataFrame(rows)

def load_excel_long(path: str, sheet: str | int | None):
    """
    Supported Excel layouts:

    A) Already long:
       columns = testimonial_id, concept, rater, score

    B) Semi-long with two model columns:
       columns = id (or testimonial_id), concept, openai, anthropic
       -> melted to long
    """
    df = pd.read_excel(path, sheet_name=sheet)
    cols = {c.strip().lower(): c for c in df.columns if isinstance(c, str)}

    # Case A: fully long
    need = {"testimonial_id", "concept", "rater", "score"}
    if need.issubset(set(cols.keys())):
        return df.rename(columns={
            cols["testimonial_id"]: "testimonial_id",
            cols["concept"]: "concept",
            cols["rater"]: "rater",
            cols["score"]: "score",
        })

    # Case B: two model columns
    if {"concept", "openai", "anthropic"}.issubset(set(cols.keys())) and \
       (("testimonial_id" in cols) or ("id" in cols)):
        id_col = cols.get("testimonial_id", cols.get("id"))
        df2 = df.rename(columns={
            id_col: "testimonial_id",
            cols["concept"]: "concept",
            cols["openai"]: "openai",
            cols["anthropic"]: "anthropic",
        }).copy()
        dfm = df2.melt(
            id_vars=["testimonial_id", "concept"],
            value_vars=["openai", "anthropic"],
            var_name="rater",
            value_name="score"
        )
        return dfm

    raise ValueError(
        "Excel not recognized. Expect either long (testimonial_id, concept, rater, score) "
        "or semi-long (id/testimonial_id, concept, openai, anthropic)."
    )

# ----------------------------
# Matrix builder
# ----------------------------

def build_matrices(long_df: pd.DataFrame):
    long_df = long_df.copy()
    long_df["testimonial_id"] = long_df["testimonial_id"].astype(str).str.strip()
    long_df["concept"] = long_df["concept"].astype(str).str.strip()
    long_df["rater"] = long_df["rater"].astype(str).str.strip()

    testimonials = sorted(long_df["testimonial_id"].unique().tolist())
    raters = sorted(long_df["rater"].unique().tolist())
    t_index = {t: i for i, t in enumerate(testimonials)}
    r_index = {r: j for j, r in enumerate(raters)}

    mats = {}
    for concept, sub in long_df.groupby("concept", sort=True):
        M = np.full((len(testimonials), len(raters)), np.nan, dtype=float)
        for _, row in sub.iterrows():
            i = t_index[row["testimonial_id"]]
            j = r_index[row["rater"]]
            M[i, j] = float(row["score"])
        mats[concept] = M
    return mats, testimonials, raters

# ----------------------------
# Writers: JSON + Excel
# ----------------------------

def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def to_record_list(df: pd.DataFrame):
    return json.loads(df.to_json(orient="records"))

def write_excel(out_path: str, tables: dict[str, pd.DataFrame]):
    # Requires openpyxl
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for sheet, df in tables.items():
            df.to_excel(xw, sheet_name=sheet, index=False)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Reliability: Krippendorff α + Interval α + ICC(2,1)/(2,k) from JSON or Excel.\n"
                    "Defaults let you pass only --input."
    )
    ap.add_argument("--input", required=True, help="Path to .json/.jsonl/.jl or .xlsx/.xls")
    # All other args have sensible defaults; override only if you want.
    ap.add_argument("--sheet", default=None, help=f"Excel sheet (default: first sheet index {DEFAULT_SHEET_INDEX})")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Alpha binarization threshold")
    ap.add_argument("--boots", type=int, default=DEFAULT_BOOTS, help="Bootstrap reps for alpha/interval-alpha CIs")
    ap.add_argument("--perms", type=int, default=DEFAULT_PERMS, help="Permutation reps for alpha>0 p-value")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Alpha level for analytical ICC CI")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    ap.add_argument("--format", choices=["json", "excel", "both"], default=DEFAULT_FORMAT,
                    help="Output format(s)")
    ap.add_argument("--excel_name", default="results.xlsx", help="Excel filename")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load
    ext = pathlib.Path(args.input).suffix.lower()
    if ext in (".json", ".jsonl", ".jl"):
        records = load_json_any(args.input)
        df_long = json_to_long(records)
    elif ext in (".xlsx", ".xls"):
        # If user didn't provide a sheet, use first sheet
        sheet = args.sheet
        if sheet is None:
            sheet = DEFAULT_SHEET_INDEX
        else:
            # allow numeric or name
            try:
                sheet = int(sheet)
            except:
                pass
        df_long = load_excel_long(args.input, sheet=sheet)
    else:
        raise ValueError("Unsupported input type. Use .json/.jsonl/.jl or .xlsx/.xls")

    # Build matrices
    continuous_by_concept, testimonials, raters = build_matrices(df_long)

    # (1) Nominal Alpha per concept (binary) + CIs + prevalence + n_items
    alpha_rows = []
    binary_by_concept = {}
    for concept, mat in continuous_by_concept.items():
        bin_mat = threshold_continuous_to_binary(mat, threshold=args.threshold)
        binary_by_concept[concept] = bin_mat

        a_hat, lo, hi = bootstrap_alpha_nominal(bin_mat, n_boot=args.boots, random_state=42)
        a_hat, lo, hi = _clip_01(a_hat), _clip_01(lo), _clip_01(hi)

        # rows (items) with at least 2 valid ratings
        n_items_alpha = int(((~np.isnan(bin_mat)).sum(axis=1) >= 2).sum())

        prev = _prevalence(bin_mat)
        prev = _clip_01(prev) if prev == prev else prev  # clip if not NaN

        row = {
            "concept": concept,
            "alpha": a_hat,
            "alpha_ci_low": lo,
            "alpha_ci_high": hi,
            "alpha_flag": _alpha_flag(a_hat, lo, hi),
            "prevalence": prev,
            "n_items_alpha": n_items_alpha,
        }
        if args.perms and args.perms > 0:
            p = permutation_test_alpha_nominal(bin_mat, n_perm=args.perms, random_state=42)
            row["alpha_p_perm"] = p
        alpha_rows.append(row)
    df_alpha = pd.DataFrame(alpha_rows).sort_values("concept")

    # (1b) Overall alpha (binary)
    a_overall, lo_overall, hi_overall = bootstrap_overall_alpha_from_concepts(
        binary_by_concept, n_boot=args.boots, random_state=42
    )
    alpha_overall_obj = {
        "alpha_overall": _clip_01(a_overall),
        "ci_low": _clip_01(lo_overall),
        "ci_high": _clip_01(hi_overall),
    }

    # (1c) Interval Alpha per concept + CIs (no thresholding)
    alpha_int_rows = []
    for concept, mat in continuous_by_concept.items():
        a_pt, a_lo, a_hi = bootstrap_alpha_interval(mat, n_boot=args.boots, random_state=42)
        a_pt, a_lo, a_hi = _clip_01(a_pt), _clip_01(a_lo), _clip_01(a_hi)

        n_items_alpha_int = int(((~np.isnan(mat)).sum(axis=1) >= 2).sum())

        alpha_int_rows.append({
            "concept": concept,
            "alpha_interval": a_pt,
            "alpha_interval_ci_low": a_lo,
            "alpha_interval_ci_high": a_hi,
            "n_items_alpha_interval": n_items_alpha_int,
        })
    df_alpha_interval = pd.DataFrame(alpha_int_rows).sort_values("concept")

    # (1d) Overall Interval Alpha
    a_int_overall, a_int_lo, a_int_hi = bootstrap_overall_alpha_interval_from_concepts(
        continuous_by_concept, n_boot=args.boots, random_state=42
    )
    alpha_interval_overall_obj = {
        "alpha_interval_overall": _clip_01(a_int_overall),
        "ci_low": _clip_01(a_int_lo),
        "ci_high": _clip_01(a_int_hi),
    }

    # (2) ICC per concept (point + analytical CI if SciPy present)
    icc_rows = []
    for concept, mat in continuous_by_concept.items():
        base = icc_2_1_and_2_k(mat)
        try:
            ci = icc_analytic_ci(mat, alpha=args.alpha)  # needs SciPy
            ICC2_1, ICC2_1_lo, ICC2_1_hi = ci["ICC2_1"]
            ICC2_k, ICC2_k_lo, ICC2_k_hi = ci["ICC2_k"]
        except ImportError:
            ICC2_1, ICC2_1_lo, ICC2_1_hi = base["ICC2_1"], np.nan, np.nan
            ICC2_k, ICC2_k_lo, ICC2_k_hi = base["ICC2_k"], np.nan, np.nan

        icc_rows.append({
            "concept": concept,
            "n_items_used": base["n_items_used"],
            "n_raters": base["n_raters"],
            "ICC2_1": ICC2_1,
            "ICC2_1_ci_low": ICC2_1_lo,
            "ICC2_1_ci_high": ICC2_1_hi,
            "ICC2_1_band": _icc_band(ICC2_1),
            "ICC2_k": ICC2_k,
            "ICC2_k_ci_low": ICC2_k_lo,
            "ICC2_k_ci_high": ICC2_k_hi,
            "ICC2_k_band": _icc_band(ICC2_k),
        })
    df_icc = pd.DataFrame(icc_rows).sort_values("concept")

    # Optional overall ICC (stacked rows)
    all_rows = []
    for concept, mat in continuous_by_concept.items():
        all_rows.extend([mat[i, :] for i in range(mat.shape[0])])
    all_mat = np.vstack(all_rows)
    base_all = icc_2_1_and_2_k(all_mat)
    try:
        ci_all = icc_analytic_ci(all_mat, alpha=args.alpha)
        icc_overall_obj = {
            "overall_ICC2_1": ci_all["ICC2_1"][0], "overall_ICC2_1_ci_low": ci_all["ICC2_1"][1], "overall_ICC2_1_ci_high": ci_all["ICC2_1"][2],
            "overall_ICC2_k": ci_all["ICC2_k"][0], "overall_ICC2_k_ci_low": ci_all["ICC2_k"][1], "overall_ICC2_k_ci_high": ci_all["ICC2_k"][2],
        }
    except ImportError:
        icc_overall_obj = {
            "overall_ICC2_1": base_all["ICC2_1"], "overall_ICC2_1_ci_low": None, "overall_ICC2_1_ci_high": None,
            "overall_ICC2_k": base_all["ICC2_k"], "overall_ICC2_k_ci_low": None, "overall_ICC2_k_ci_high": None,
        }

    # ----------------------------
    # Write outputs (JSON / Excel)
    # ----------------------------
    os.makedirs(args.outdir, exist_ok=True)
    if args.format in ("json", "both"):
        write_json(os.path.join(args.outdir, "alpha_per_concept.json"), to_record_list(df_alpha))
        write_json(os.path.join(args.outdir, "alpha_overall.json"), alpha_overall_obj)
        write_json(os.path.join(args.outdir, "alpha_interval_per_concept.json"), to_record_list(df_alpha_interval))
        write_json(os.path.join(args.outdir, "alpha_interval_overall.json"), alpha_interval_overall_obj)
        write_json(os.path.join(args.outdir, "icc_per_concept.json"), to_record_list(df_icc))
        write_json(os.path.join(args.outdir, "icc_overall.json"), icc_overall_obj)

    if args.format in ("excel", "both"):
        # Requires openpyxl
        tables = {
            "alpha_per_concept": df_alpha,
            "alpha_overall": pd.DataFrame([alpha_overall_obj]),
            "alpha_interval_per_concept": df_alpha_interval,
            "alpha_interval_overall": pd.DataFrame([alpha_interval_overall_obj]),
            "icc_per_concept": df_icc,
            "icc_overall": pd.DataFrame([icc_overall_obj]),
        }
        write_excel(os.path.join(args.outdir, args.excel_name), tables)

    print("Done. Wrote to:", os.path.abspath(args.outdir))
    if args.format in ("excel", "both"):
        print("Excel:", os.path.join(args.outdir, args.excel_name))

if __name__ == "__main__":
    main()
