# reliability_metrics.py
# ------------------------------------------------------------
# Krippendorff's alpha (nominal) with cluster-bootstrap CIs and permutation p
# ICC(2,1) and ICC(2,k) with ANALYTICAL 95% CIs (Shrout–Fleiss / McGraw–Wong).
# ------------------------------------------------------------
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

# ====== NEW: optional import for interval alpha ======
try:
    import krippendorff  # pip install krippendorff
    _HAS_KRIPP = True
except Exception:
    _HAS_KRIPP = False

# -------- Helpers (NEW) --------
def _mask_rows(X: np.ndarray, require_all_raters: bool = False, min_raters: int = 2) -> np.ndarray:
    """
    Returns a boolean mask of rows/items to keep.
    - require_all_raters=True: keep only complete rows (for ICC etc.)
    - else: keep rows with at least `min_raters` finite ratings (for alpha)
    """
    if require_all_raters:
        return ~np.isnan(X).any(axis=1)
    counts = (~np.isnan(X)).sum(axis=1)
    return counts >= min_raters

def _percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo, hi = np.percentile(samples, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

# -------- Krippendorff's alpha (nominal) - YOUR CODE (unchanged) --------

def _coincidence_matrix_nominal(ratings: np.ndarray) -> Tuple[np.ndarray, dict]:
    observed = np.unique(ratings[~np.isnan(ratings)])
    cat_to_idx = {cat: i for i, cat in enumerate(observed)}
    k = len(observed)
    C = np.zeros((k, k), dtype=float)

    for i in range(ratings.shape[0]):
        row = ratings[i, :]
        vals = row[~np.isnan(row)]
        m = len(vals)
        if m < 2:
            continue
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        cats = list(counts.keys())
        for a in cats:
            ia = cat_to_idx[a]
            na = counts[a]
            C[ia, ia] += na * (na - 1)
            for b in cats:
                if a == b:
                    continue
                ib = cat_to_idx[b]
                C[ia, ib] += na * counts[b]
    return C, cat_to_idx

def krippendorff_alpha_nominal(ratings: np.ndarray) -> float:
    if ratings.ndim != 2:
        raise ValueError("ratings must be 2D")
    valid_pairs = sum(np.sum(~np.isnan(ratings[i, :])) >= 2 for i in range(ratings.shape[0]))
    if valid_pairs == 0:
        return np.nan

    C, _ = _coincidence_matrix_nominal(ratings)
    total_pairs = C.sum()
    if total_pairs == 0:
        return np.nan

    Do = (total_pairs - np.trace(C)) / total_pairs
    n_c = C.sum(axis=1)
    N = n_c.sum()
    if N <= 1:
        return np.nan
    De = 1.0 - np.sum(n_c * (n_c - 1.0)) / (N * (N - 1.0))
    if De == 0:
        return 1.0 if Do == 0 else np.nan
    return 1.0 - (Do / De)

def bootstrap_alpha_nominal(
    ratings: np.ndarray, n_boot: int = 5000, random_state: Optional[int] = None, alpha: float = 0.05
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    n_items = ratings.shape[0]
    alpha_hat = krippendorff_alpha_nominal(ratings)
    alphas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)  # cluster bootstrap over items
        boot = ratings[idx, :]
        a = krippendorff_alpha_nominal(boot)
        if not np.isnan(a):
            alphas.append(a)
    if len(alphas) == 0:
        return alpha_hat, np.nan, np.nan
    lo, hi = _percentile_ci(np.array(alphas), alpha=alpha)
    return float(alpha_hat), float(lo), float(hi)

def permutation_test_alpha_nominal(
    ratings: np.ndarray, n_perm: int = 5000, random_state: Optional[int] = None
) -> float:
    rng = np.random.default_rng(random_state)
    alpha_obs = krippendorff_alpha_nominal(ratings)
    if np.isnan(alpha_obs):
        return np.nan
    n_items, n_raters = ratings.shape
    perm_alphas = []
    for _ in range(n_perm):
        perm = ratings.copy()
        for j in range(n_raters):
            col = perm[:, j]
            mask = ~np.isnan(col)
            perm[mask, j] = rng.permutation(col[mask])
        a = krippendorff_alpha_nominal(perm)
        if not np.isnan(a):
            perm_alphas.append(a)
    perm_alphas = np.array(perm_alphas)
    return float((np.sum(perm_alphas >= alpha_obs) + 1.0) / (len(perm_alphas) + 1.0))

# -------- NEW: Krippendorff's alpha (interval) + bootstrap CI --------

def alpha_interval_point(ratings_continuous: np.ndarray) -> float:
    """
    ratings_continuous: (n_items, n_raters) floats in [0,1], np.nan allowed.
    Uses krippendorff package. Returns NaN if unavailable.
    """
    if not _HAS_KRIPP:
        return np.nan
    # Option 1 (full data): allow missingness (>=2 raters/item)
    mask = _mask_rows(ratings_continuous, require_all_raters=False, min_raters=2)
    X = ratings_continuous[mask]
    if X.shape[0] < 2:
        return np.nan
    return float(krippendorff.alpha(X.T, level_of_measurement="interval"))

def bootstrap_alpha_interval(
    ratings_continuous: np.ndarray, n_boot: int = 5000, random_state: Optional[int] = None, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap over items (same mask as point estimate).
    """
    if not _HAS_KRIPP:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(random_state)
    mask = _mask_rows(ratings_continuous, require_all_raters=False, min_raters=2)
    X = ratings_continuous[mask]
    n = X.shape[0]
    if n < 2:
        return np.nan, np.nan, np.nan

    def _a(arr): return krippendorff.alpha(arr.T, level_of_measurement="interval")

    point = float(_a(X))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample items
        boots.append(float(_a(X[idx])))
    lo, hi = _percentile_ci(np.array(boots), alpha=alpha)
    return point, float(lo), float(hi)

# -------- ICC(2,1) and ICC(2,k) + Analytical CIs (YOUR CODE) --------

def _anova_components_complete(ratings: np.ndarray):
    X = ratings.copy()
    mask = ~np.isnan(X).any(axis=1)  # complete-case rows for ICC
    X = X[mask, :]
    n, k = X.shape
    if n < 2 or k < 2:
        return np.nan, np.nan, np.nan, 0, 0, 0
    mean_per_item = X.mean(axis=1, keepdims=True)
    mean_per_rater = X.mean(axis=0, keepdims=True)
    grand_mean = X.mean()
    ss_items = k * np.sum((mean_per_item - grand_mean) ** 2)
    ss_raters = n * np.sum((mean_per_rater - grand_mean) ** 2)
    ss_total = np.sum((X - grand_mean) ** 2)
    ss_error = ss_total - ss_items - ss_raters
    df_items = n - 1
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)
    ms_items = ss_items / df_items
    ms_raters = ss_raters / df_raters
    ms_error = ss_error / df_error
    return (ms_items, ms_raters, ms_error, n, k, mask.sum())

def icc_2_1_and_2_k(ratings: np.ndarray) -> Dict[str, float]:
    ms_items, ms_raters, ms_error, n, k, n_used = _anova_components_complete(ratings)
    if n_used == 0 or np.isnan(ms_items):
        return {'ICC2_1': np.nan, 'ICC2_k': np.nan, 'n_items_used': 0, 'n_raters': ratings.shape[1]}
    icc2_1 = (ms_items - ms_error) / (ms_items + (k - 1) * ms_error + (k * (ms_raters - ms_error) / n))
    icc2_k = (ms_items - ms_error) / (ms_items + (ms_raters - ms_error) / n)
    return {
        'ICC2_1': float(icc2_1),
        'ICC2_k': float(icc2_k),
        'n_items_used': int(n),
        'n_raters': int(k),
        'MS_items': float(ms_items),
        'MS_raters': float(ms_raters),
        'MS_error': float(ms_error),
        'DF_items': int(n - 1),
        'DF_raters': int(k - 1),
        'DF_error': int((n - 1) * (k - 1)),
    }

def icc_analytic_ci(
    ratings: np.ndarray, alpha: float = 0.05
) -> Dict[str, Tuple[float, float, float]]:
    """
    Analytical 95% CIs for ICC(2,1) and ICC(2,k) using F-quantiles.
    """
    from math import isfinite
    try:
        from scipy.stats import f as f_dist
    except Exception as e:
        raise ImportError("SciPy is required for analytical ICC CIs") from e

    stats = icc_2_1_and_2_k(ratings)
    icc2_1 = stats['ICC2_1']; icc2_k = stats['ICC2_k']
    ms_items = stats['MS_items']; ms_raters = stats['MS_raters']; ms_error = stats['MS_error']
    n = stats['n_items_used']; k = stats['n_raters']
    if n < 2 or k < 2 or not all(isfinite(x) for x in [ms_items, ms_raters, ms_error]):
        return {'ICC2_1': (np.nan, np.nan, np.nan), 'ICC2_k': (np.nan, np.nan, np.nan)}

    F = ms_items / ms_error
    df1 = n - 1
    df2 = n * (k - 1)

    F_low = F / f_dist.ppf(1 - alpha/2, df1, df2)
    F_up  = F * f_dist.ppf(1 - alpha/2, df2, df1)

    m = ms_raters / ms_error
    adj = (k - 1) + (k / n) * (m - 1)

    icc2_1_low = (F_low - 1) / (F_low + adj)
    icc2_1_up  = (F_up  - 1) / (F_up  + adj)

    adj_k = (m - 1) / n
    icc2_k_low = (F_low - 1) / (F_low + adj_k)
    icc2_k_up  = (F_up  - 1) / (F_up  + adj_k)

    def clip11(x): return float(max(-1.0, min(1.0, x)))
    return {
        'ICC2_1': (float(icc2_1), clip11(icc2_1_low), clip11(icc2_1_up)),
        'ICC2_k': (float(icc2_k), clip11(icc2_k_low), clip11(icc2_k_up)),
    }

# -------- Helpers you already had --------

def threshold_continuous_to_binary(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    out = scores.copy().astype(float)
    mask = ~np.isnan(out)
    out[mask] = (out[mask] >= threshold).astype(float)
    return out

def stack_concepts_for_overall_alpha(concept_mats: Dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([concept_mats[k] for k in sorted(concept_mats.keys())])

def bootstrap_overall_alpha_from_concepts(
    concept_mats: Dict[str, np.ndarray], n_boot: int = 5000, random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    first = next(iter(concept_mats))
    n_items = concept_mats[first].shape[0]
    stacked = stack_concepts_for_overall_alpha(concept_mats)
    point = krippendorff_alpha_nominal(stacked)
    alphas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        boot_stacked = np.vstack([mat[idx, :] for mat in concept_mats.values()])
        a = krippendorff_alpha_nominal(boot_stacked)
        if not np.isnan(a):
            alphas.append(a)
    if not alphas:
        return point, np.nan, np.nan
    lo, hi = _percentile_ci(np.array(alphas))
    return float(point), float(lo), float(hi)

def bootstrap_overall_alpha_interval_from_concepts(
    concept_mats: Dict[str, np.ndarray],
    n_boot: int = 5000,
    random_state: Optional[int] = None,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    if not _HAS_KRIPP:
        return np.nan, np.nan, np.nan
    import numpy as np
    rng = np.random.default_rng(random_state)
    first = next(iter(concept_mats))
    n_items = concept_mats[first].shape[0]

    def _alpha(arr):  # arr: (items, raters)
        return float(krippendorff.alpha(arr.T, level_of_measurement="interval"))

    stacked = np.vstack([concept_mats[k] for k in concept_mats.keys()])
    point = _alpha(stacked)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        boot_stacked = np.vstack([mat[idx, :] for mat in concept_mats.values()])
        boots.append(_alpha(boot_stacked))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)

# -------- NEW: one-call convenience wrapper per concept --------
def compute_reliability_for_concept(
    X_continuous: np.ndarray,  # (items, raters) float in [0,1], np.nan allowed
    binarize_threshold: float = 0.50,
    n_boot: int = 5000,
    seed: Optional[int] = 42
) -> Dict[str, object]:
    """
    Returns point estimates + 95% CIs for:
      - alpha_interval (primary)
      - ICC(2,1) and ICC(2,k) (co-primary)
      - alpha_nominal at the given threshold (sensitivity)
    Uses the SAME item set per metric family:
      - alpha_interval: rows with >=2 raters
      - ICC: complete rows only
      - alpha_nominal: rows with >=2 raters after thresholding
    """
    # α_interval
    a_int, a_int_lo, a_int_hi = bootstrap_alpha_interval(X_continuous, n_boot, seed)

    # ICC on complete rows
    icc_stats = icc_analytic_ci(X_continuous)
    icc2_1_point, icc2_1_lo, icc2_1_hi = icc_stats['ICC2_1']
    icc2_k_point, icc2_k_lo, icc2_k_hi = icc_stats['ICC2_k']

    # α_nominal sensitivity
    X_bin = threshold_continuous_to_binary(X_continuous, binarize_threshold)
    a_nom, a_nom_lo, a_nom_hi = bootstrap_alpha_nominal(X_bin, n_boot, seed)

    return {
        'alpha_interval': (a_int, a_int_lo, a_int_hi),
        'ICC2_1': (icc2_1_point, icc2_1_lo, icc2_1_hi),
        'ICC2_k': (icc2_k_point, icc2_k_lo, icc2_k_hi),
        'alpha_nominal': (a_nom, a_nom_lo, a_nom_hi),
        'threshold': binarize_threshold,
        'n_items_alpha_interval': int(_mask_rows(X_continuous, False, 2).sum()),
        'n_items_icc_complete': int(_mask_rows(X_continuous, True).sum()),
        'n_items_alpha_nominal': int(_mask_rows(X_bin, False, 2).sum()),
    }

