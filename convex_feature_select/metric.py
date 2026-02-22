"""
convex_feature_select.metric
============================
Implementation of the Δ_CH dissimilarity metric and subset ranking.

The metric is defined in Equation (9) of the paper:

    Δ_CH(S) = (1 / n_d) * Σ_P  1[ Σ_i HV_i^S(P) == 1 ]

where HV_i^S(P) = 1 if point P lies inside the convex hull of class i
under feature subset S, and 0 otherwise.  A point is "non-confusing" if
it lies inside exactly one hull; the metric counts the fraction of such
points.

Computational notes
-------------------
* Convex hull construction uses SciPy's QHull (Quickhull algorithm),
  O(n log n) average complexity.
* Point-in-hull queries use Delaunay triangulation, O(n log n) average.
* In 1-D a hull degenerates to an interval [min, max]; this is handled
  explicitly to avoid QHull errors.
* Degenerate / collinear point clouds (rank-deficient) are handled by
  catching QhullError and treating all points as confusing for that hull.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError


__all__ = ["delta_ch", "hull_vectors", "rank_subsets"]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def delta_ch(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Sequence[int] | None = None,
) -> float:
    """Compute the Δ_CH dissimilarity metric for a feature subset.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data (all features).
    y : array-like, shape (n_samples,)
        Class labels.
    feature_indices : sequence of int, optional
        Indices of the features that form the subset S.
        If ``None``, all features are used.

    Returns
    -------
    float
        Δ_CH ∈ [0, 1].  Higher means better class separation.

    Examples
    --------
    >>> import numpy as np
    >>> from convex_feature_select import delta_ch
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal([0, 0], 0.3, (50, 2)),
    ...                rng.normal([2, 2], 0.3, (50, 2))])
    >>> y = np.array([0]*50 + [1]*50)
    >>> round(delta_ch(X, y), 2)
    1.0
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)

    if feature_indices is not None:
        X_sub = X_arr[:, list(feature_indices)]
    else:
        X_sub = X_arr

    hv = hull_vectors(X_sub, y_arr)            # (n_samples, n_classes)
    non_confusing = (hv.sum(axis=1) == 1)
    return float(non_confusing.mean())


def hull_vectors(
    X_sub: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Compute hull vectors for all data points.

    For each data point and each class, the hull vector element is 1 if
    the point lies inside (or on the boundary of) the class convex hull,
    and 0 otherwise.

    Parameters
    ----------
    X_sub : array, shape (n_samples, n_features_sub)
        Data projected onto the feature subset of interest.
    y : array, shape (n_samples,)
        Class labels.

    Returns
    -------
    hv : np.ndarray, shape (n_samples, n_classes)
        Binary hull-membership matrix.
    """
    classes = np.unique(y)
    n_samples = len(X_sub)
    n_classes = len(classes)
    hv = np.zeros((n_samples, n_classes), dtype=np.int8)

    for col, cls in enumerate(classes):
        mask = (y == cls)
        hull_pts = X_sub[mask]
        hv[:, col] = _points_in_hull(X_sub, hull_pts)

    return hv


def rank_subsets(
    X: np.ndarray,
    y: np.ndarray,
    n_features_to_select: int,
    *,
    max_subsets: int | None = None,
) -> list[tuple[tuple[int, ...], float]]:
    """Rank all C(n_features, n_features_to_select) subsets by Δ_CH.

    Based on the monotonicity property (Equation 10), only subsets of
    exactly ``n_features_to_select`` features need to be evaluated —
    smaller subsets always have Δ_CH ≤ larger subsets, so they can be
    safely pruned.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Class labels.
    n_features_to_select : int
        Size of each feature subset (cardinality constraint k).
    max_subsets : int, optional
        Maximum number of subsets to evaluate (useful for large d).
        If ``None``, all C(n_features, n_features_to_select) are evaluated.
        When set, subsets are drawn in lexicographic order.

    Returns
    -------
    ranked : list of (feature_indices_tuple, delta_ch_score)
        Sorted descending by Δ_CH score.

    Examples
    --------
    >>> import numpy as np
    >>> from convex_feature_select import rank_subsets
    >>> rng = np.random.default_rng(1)
    >>> X = np.hstack([rng.normal([0, 2], 0.3, (100, 2)),   # good features
    ...                rng.normal(0, 1, (100, 2))])           # noise features
    >>> y = np.array([0]*50 + [1]*50)
    >>> ranked = rank_subsets(X, y, n_features_to_select=2)
    >>> ranked[0][0]   # best 2-feature subset should include features 0 and 1
    (0, 1)
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    n_features = X_arr.shape[1]

    all_combos = list(combinations(range(n_features), n_features_to_select))
    if max_subsets is not None:
        all_combos = all_combos[:max_subsets]

    results = [
        (combo, delta_ch(X_arr, y_arr, combo))
        for combo in all_combos
    ]
    results.sort(key=lambda t: t[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _points_in_hull(query_pts: np.ndarray, hull_pts: np.ndarray) -> np.ndarray:
    """Return a binary array: 1 if query point is inside the convex hull of
    hull_pts, 0 otherwise.

    Handles special cases:
    * 1-D: hull is the interval [min, max].
    * Degenerate clouds (rank-deficient, all points collinear, etc.):
      uses interval/bounding-box fallback.
    * Single point: only that exact point is "inside".
    """
    n_q = len(query_pts)
    result = np.zeros(n_q, dtype=np.int8)

    if len(hull_pts) == 0:
        return result

    d = hull_pts.shape[1]

    # --- 1-D special case ---------------------------------------------------
    if d == 1:
        lo, hi = hull_pts[:, 0].min(), hull_pts[:, 0].max()
        q = query_pts[:, 0]
        result[:] = ((q >= lo) & (q <= hi)).astype(np.int8)
        return result

    # --- single point --------------------------------------------------------
    if len(hull_pts) == 1:
        eq = np.all(np.isclose(query_pts, hull_pts[0]), axis=1)
        result[eq] = 1
        return result

    # --- general case via Delaunay triangulation ----------------------------
    try:
        tri = Delaunay(hull_pts)
        inside = tri.find_simplex(query_pts) >= 0
        result[:] = inside.astype(np.int8)
    except QhullError:
        # Degenerate cloud (collinear / rank-deficient): project to 1-D via PCA
        centered = hull_pts - hull_pts.mean(axis=0)
        cov = centered.T @ centered
        try:
            _, vecs = np.linalg.eigh(cov)
            axis = vecs[:, -1]                     # principal axis
            proj_hull = hull_pts @ axis
            proj_q    = query_pts @ axis
            lo, hi = proj_hull.min(), proj_hull.max()
            result[:] = ((proj_q >= lo) & (proj_q <= hi)).astype(np.int8)
        except Exception:
            pass  # leave as zeros (all confusing)

    return result
