"""
convex_feature_select
=====================
A scikit-learn compatible Python package for two-step, convex hull-based
feature selection for classification tasks.

The package implements the methodology proposed in:

    Du, D., Karve, P., & Mahadevan, S. (2025).
    "Feature selection for classification models using feature space geometric
    structure revealed by convex hulls."
    (under review)

Core idea
---------
**Step 1 – Δ_CH screening (filter step)**
    For each candidate feature subset S, construct a convex hull per class in
    the feature subspace.  A data point is "confusing" if it lies inside more
    than one hull (i.e., the class boundaries overlap at that point).
    The dissimilarity metric is defined as::

        Δ_CH(S) = (number of non-confusing points) / (total points)

    Higher Δ_CH → classes are better separated → S is a more promising subset.
    The top-m subsets are retained.

**Step 2 – Exhaustive search over top-m subsets (wrapper step)**
    Train user-supplied classification model(s) for each of the top-m subsets
    and select the one with the highest cross-validated accuracy.

The two-step approach achieves accuracy comparable to exhaustive wrapper
search at a fraction of the computational cost.

Public API
----------
ConvexHullFeatureSelector   – main sklearn-compatible estimator
delta_ch                    – compute Δ_CH for a single feature subset
rank_subsets                – rank all candidate subsets by Δ_CH

Reference
---------
Du, D., Karve, P., & Mahadevan, S. (2025). Feature selection for
classification models using feature space geometric structure revealed by
convex hulls. (under review)
"""

from .metric  import delta_ch, rank_subsets
from .selector import ConvexHullFeatureSelector

__all__ = [
    "ConvexHullFeatureSelector",
    "delta_ch",
    "rank_subsets",
]

__version__ = "0.1.0"
