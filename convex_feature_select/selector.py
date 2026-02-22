"""
convex_feature_select.selector
==============================
Scikit-learn compatible estimator implementing the two-step feature
selection methodology.

The estimator follows the standard sklearn API:

    selector = ConvexHullFeatureSelector(
        n_features_to_select=3,
        top_m=50,
        estimator=RandomForestClassifier(),
        cv=5,
    )
    selector.fit(X_train, y_train)
    X_reduced = selector.transform(X_train)

It also supports ``set_output(transform="pandas")`` if pandas is installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

from .metric import delta_ch, rank_subsets


__all__ = ["ConvexHullFeatureSelector"]


class ConvexHullFeatureSelector(TransformerMixin, BaseEstimator):
    """Two-step convex hull-based feature selector for classification.

    **Step 1 – Δ_CH screening**
        All C(n_features, n_features_to_select) feature subsets of the
        specified size are ranked by the Δ_CH dissimilarity metric.  The
        top ``top_m`` subsets are shortlisted.

    **Step 2 – Exhaustive search over top-m subsets**
        Each shortlisted subset is evaluated by cross-validating
        ``estimator`` on the training data projected onto that subset.
        The subset with the highest mean CV score is selected.

    Parameters
    ----------
    n_features_to_select : int
        Number of features in the selected subset (cardinality constraint k).
    top_m : int, default=50
        Number of top-ranked subsets (by Δ_CH) to pass to Step 2.
        Set to 1 to skip Step 2 entirely and use only Δ_CH ranking.
    estimator : sklearn classifier, optional
        Classifier used in Step 2.  Must implement ``fit`` and ``predict``.
        If ``None``, Step 2 is skipped and the top-1 Δ_CH subset is chosen.
    cv : int, default=5
        Number of cross-validation folds used in Step 2.
    scoring : str, default='accuracy'
        Scoring metric passed to ``cross_val_score`` in Step 2.
    max_subsets : int, optional
        Cap on the total number of subsets evaluated in Step 1.
        Useful when n_features is large.  If ``None``, all subsets are used.
    n_jobs : int, default=1
        Parallel jobs for cross-validation in Step 2.
        Pass ``-1`` to use all available cores.
    verbose : int, default=0
        Verbosity level (0 = silent, 1 = progress, 2 = detailed).

    Attributes
    ----------
    selected_features_ : tuple of int
        Indices of the selected features after fitting.
    delta_ch_score_ : float
        Δ_CH value of the selected feature subset.
    cv_score_ : float or None
        Mean cross-validated accuracy of the selected subset (None if
        Step 2 was skipped).
    ranking_ : list of (tuple, float)
        Full ranking of all evaluated subsets by Δ_CH (Step 1 output).
    n_features_in_ : int
        Total number of features seen during fit.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from convex_feature_select import ConvexHullFeatureSelector
    >>>
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> selector = ConvexHullFeatureSelector(
    ...     n_features_to_select=3,
    ...     top_m=50,
    ...     estimator=RandomForestClassifier(n_estimators=100, random_state=0),
    ...     cv=5,
    ... )
    >>> selector.fit(X, y)
    ConvexHullFeatureSelector(...)
    >>> selector.selected_features_
    (...)
    >>> X_reduced = selector.transform(X)
    >>> X_reduced.shape
    (569, 3)
    """

    def __init__(
        self,
        n_features_to_select: int = 3,
        top_m: int = 50,
        estimator: Any = None,
        cv: int = 5,
        scoring: str = "accuracy",
        max_subsets: int | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ):
        self.n_features_to_select = n_features_to_select
        self.top_m                = top_m
        self.estimator            = estimator
        self.cv                   = cv
        self.scoring              = scoring
        self.max_subsets          = max_subsets
        self.n_jobs               = n_jobs
        self.verbose              = verbose

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConvexHullFeatureSelector":
        """Fit the selector by running the two-step selection procedure.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        self.n_features_in_ = X_arr.shape[1]

        self._validate_params(X_arr)

        # ---- Step 1: rank all subsets by Δ_CH ----------------------------
        if self.verbose >= 1:
            print(f"[ConvexHullFeatureSelector] Step 1: ranking subsets by Δ_CH ...")

        self.ranking_ = rank_subsets(
            X_arr, y_arr,
            n_features_to_select=self.n_features_to_select,
            max_subsets=self.max_subsets,
        )

        top_candidates = self.ranking_[: self.top_m]

        if self.verbose >= 1:
            best_delta = top_candidates[0][1]
            print(
                f"[ConvexHullFeatureSelector] Step 1 done.  "
                f"Evaluated {len(self.ranking_)} subsets.  "
                f"Best Δ_CH = {best_delta:.4f}  "
                f"({len(top_candidates)} subsets shortlisted for Step 2)"
            )

        # ---- Step 2: exhaustive CV over top-m subsets --------------------
        if self.estimator is None or self.top_m <= 1:
            # Skip Step 2 — pick the top Δ_CH subset directly
            best_subset, best_delta = top_candidates[0]
            self.selected_features_ = best_subset
            self.delta_ch_score_    = best_delta
            self.cv_score_          = None

            if self.verbose >= 1:
                print(
                    f"[ConvexHullFeatureSelector] Step 2 skipped.  "
                    f"Selected features: {self.selected_features_}  "
                    f"(Δ_CH = {self.delta_ch_score_:.4f})"
                )
            return self

        if self.verbose >= 1:
            print(
                f"[ConvexHullFeatureSelector] Step 2: CV evaluation of "
                f"{len(top_candidates)} subsets ..."
            )

        best_subset  = None
        best_cv      = -np.inf
        best_delta_s = 0.0

        for i, (subset, delta_s) in enumerate(top_candidates):
            X_sub = X_arr[:, list(subset)]
            est   = clone(self.estimator)
            scores = cross_val_score(
                est, X_sub, y_arr,
                cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs,
            )
            mean_score = scores.mean()

            if self.verbose >= 2:
                print(
                    f"  [{i+1}/{len(top_candidates)}]  "
                    f"features={subset}  Δ_CH={delta_s:.4f}  "
                    f"CV {self.scoring}={mean_score:.4f}"
                )

            if mean_score > best_cv:
                best_cv      = mean_score
                best_subset  = subset
                best_delta_s = delta_s

        self.selected_features_ = best_subset
        self.delta_ch_score_    = best_delta_s
        self.cv_score_          = best_cv

        if self.verbose >= 1:
            print(
                f"[ConvexHullFeatureSelector] Done.  "
                f"Selected features: {self.selected_features_}  "
                f"Δ_CH = {self.delta_ch_score_:.4f}  "
                f"CV {self.scoring} = {self.cv_score_:.4f}"
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the selected feature subset.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_reduced : np.ndarray, shape (n_samples, n_features_to_select)
        """
        check_is_fitted(self, "selected_features_")
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, list(self.selected_features_)]

    def get_support(self, indices: bool = False):
        """Return a mask or indices of the selected features.

        Parameters
        ----------
        indices : bool, default=False
            If ``True``, return indices; otherwise return a boolean mask.

        Returns
        -------
        mask : np.ndarray of bool, or np.ndarray of int
        """
        check_is_fitted(self, "selected_features_")
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[list(self.selected_features_)] = True
        if indices:
            return np.where(mask)[0]
        return mask

    def get_feature_names_out(self, input_features=None):
        """Get feature names for the selected features.

        Parameters
        ----------
        input_features : array-like of str, optional
            Input feature names.  If ``None``, uses ``x0``, ``x1``, etc.

        Returns
        -------
        feature_names_out : np.ndarray of str
        """
        check_is_fitted(self, "selected_features_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.array([input_features[i] for i in self.selected_features_])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_params(self, X: np.ndarray):
        n_features = X.shape[1]
        if self.n_features_to_select >= n_features:
            raise ValueError(
                f"n_features_to_select ({self.n_features_to_select}) must be "
                f"less than the number of features ({n_features})."
            )
        if self.top_m < 1:
            raise ValueError("top_m must be >= 1.")
        if self.cv < 2:
            raise ValueError("cv must be >= 2.")

    def summary(self) -> str:
        """Return a human-readable summary of the fitted selector."""
        check_is_fitted(self, "selected_features_")
        lines = [
            "ConvexHullFeatureSelector – fit summary",
            f"  n_features_in          : {self.n_features_in_}",
            f"  n_features_to_select   : {self.n_features_to_select}",
            f"  subsets evaluated      : {len(self.ranking_)}",
            f"  top_m shortlisted      : {min(self.top_m, len(self.ranking_))}",
            f"  selected features      : {self.selected_features_}",
            f"  Δ_CH score             : {self.delta_ch_score_:.4f}",
        ]
        if self.cv_score_ is not None:
            lines.append(f"  CV {self.scoring:<18}: {self.cv_score_:.4f}")
        else:
            lines.append("  CV score               : (Step 2 skipped)")
        return "\n".join(lines)
