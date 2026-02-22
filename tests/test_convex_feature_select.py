"""
Tests for convex_feature_select
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from convex_feature_select import ConvexHullFeatureSelector, delta_ch, rank_subsets
from convex_feature_select.metric import hull_vectors, _points_in_hull


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_separable(n=100, rng=None):
    """Two perfectly separated Gaussian clusters."""
    if rng is None:
        rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([0, 0], 0.3, (n//2, 2)),
                   rng.normal([5, 5], 0.3, (n//2, 2))])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y


def make_overlapping(n=100, rng=None):
    """Two heavily overlapping Gaussian clusters."""
    if rng is None:
        rng = np.random.default_rng(1)
    X = np.vstack([rng.normal([0, 0], 2.0, (n//2, 2)),
                   rng.normal([1, 1], 2.0, (n//2, 2))])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y


# ---------------------------------------------------------------------------
# Tests: _points_in_hull
# ---------------------------------------------------------------------------

class TestPointsInHull:
    def test_point_inside(self):
        hull_pts = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
        q = np.array([[0.5, 0.5]])
        assert _points_in_hull(q, hull_pts)[0] == 1

    def test_point_outside(self):
        hull_pts = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
        q = np.array([[5.0, 5.0]])
        assert _points_in_hull(q, hull_pts)[0] == 0

    def test_1d_inside(self):
        hull_pts = np.array([[1.0],[5.0]])
        q = np.array([[3.0]])
        assert _points_in_hull(q, hull_pts)[0] == 1

    def test_1d_outside(self):
        hull_pts = np.array([[1.0],[5.0]])
        q = np.array([[0.0]])
        assert _points_in_hull(q, hull_pts)[0] == 0

    def test_single_point_hull(self):
        hull_pts = np.array([[2.0, 3.0]])
        q_match  = np.array([[2.0, 3.0]])
        q_no     = np.array([[2.1, 3.0]])
        assert _points_in_hull(q_match, hull_pts)[0] == 1
        assert _points_in_hull(q_no,    hull_pts)[0] == 0

    def test_empty_hull(self):
        hull_pts = np.zeros((0, 2))
        q = np.array([[1.0, 1.0]])
        result = _points_in_hull(q, hull_pts)
        assert result[0] == 0


# ---------------------------------------------------------------------------
# Tests: delta_ch
# ---------------------------------------------------------------------------

class TestDeltaCH:
    def test_separable_clusters_high(self):
        X, y = make_separable()
        score = delta_ch(X, y)
        assert score > 0.9, f"Expected >0.9 for separable clusters, got {score}"

    def test_overlapping_clusters_lower(self):
        X_sep, y = make_separable()
        X_ov, _  = make_overlapping()
        sep_score = delta_ch(X_sep, y)
        ov_score  = delta_ch(X_ov,  y)
        assert sep_score > ov_score

    def test_range_0_1(self):
        X, y = make_overlapping()
        score = delta_ch(X, y)
        assert 0.0 <= score <= 1.0

    def test_feature_subset(self):
        rng = np.random.default_rng(42)
        X = np.hstack([
            np.vstack([rng.normal([0,0], 0.3, (50,2)),
                       rng.normal([3,3], 0.3, (50,2))]),
            rng.normal(0, 1, (100, 3)),   # noise
        ])
        y = np.array([0]*50 + [1]*50)
        good  = delta_ch(X, y, [0, 1])
        noise = delta_ch(X, y, [2, 3])
        assert good > noise

    def test_all_features_if_no_subset(self):
        X, y = make_separable()
        assert delta_ch(X, y) == delta_ch(X, y, list(range(X.shape[1])))

    def test_multiclass(self):
        X, y = load_iris(return_X_y=True)
        score = delta_ch(X, y, [2, 3])
        assert 0.0 <= score <= 1.0

    def test_1d_feature(self):
        X = np.array([[0.], [1.], [2.], [10.], [11.], [12.]])
        y = np.array([0, 0, 0, 1, 1, 1])
        score = delta_ch(X, y)
        assert score > 0.9


# ---------------------------------------------------------------------------
# Tests: hull_vectors
# ---------------------------------------------------------------------------

class TestHullVectors:
    def test_shape(self):
        X, y = make_separable(100)
        hv = hull_vectors(X, y)
        assert hv.shape == (100, 2)

    def test_binary_values(self):
        X, y = make_separable()
        hv = hull_vectors(X, y)
        assert set(hv.ravel().tolist()).issubset({0, 1})

    def test_confusing_sum_gt_1(self):
        X, y = make_overlapping(200)
        hv = hull_vectors(X, y)
        # Some points should be confusing (sum >= 2)
        assert (hv.sum(axis=1) >= 2).any()


# ---------------------------------------------------------------------------
# Tests: rank_subsets
# ---------------------------------------------------------------------------

class TestRankSubsets:
    def test_returns_sorted(self):
        X, y = load_iris(return_X_y=True)
        ranking = rank_subsets(X, y, n_features_to_select=2)
        scores = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_correct_subset_size(self):
        X, y = load_iris(return_X_y=True)
        ranking = rank_subsets(X, y, n_features_to_select=2)
        for subset, _ in ranking:
            assert len(subset) == 2

    def test_number_of_subsets(self):
        # iris has 4 features; C(4,2)=6 subsets of size 2
        X, y = load_iris(return_X_y=True)
        ranking = rank_subsets(X, y, n_features_to_select=2)
        assert len(ranking) == 6

    def test_max_subsets_cap(self):
        X, y = load_iris(return_X_y=True)
        ranking = rank_subsets(X, y, n_features_to_select=2, max_subsets=3)
        assert len(ranking) == 3

    def test_informative_features_top_ranked(self):
        """Informative features should outrank noise features."""
        rng = np.random.default_rng(7)
        X = np.hstack([
            np.vstack([rng.normal([0,0], 0.3, (50,2)),
                       rng.normal([4,4], 0.3, (50,2))]),
            rng.normal(0, 1, (100, 2)),
        ])
        y = np.array([0]*50 + [1]*50)
        ranking = rank_subsets(X, y, n_features_to_select=2)
        top_feat = ranking[0][0]
        assert set(top_feat).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests: ConvexHullFeatureSelector
# ---------------------------------------------------------------------------

class TestConvexHullFeatureSelector:
    def test_fit_returns_self(self):
        X, y = load_iris(return_X_y=True)
        sel = ConvexHullFeatureSelector(n_features_to_select=2, top_m=3,
                                        estimator=None)
        assert sel.fit(X, y) is sel

    def test_transform_shape(self):
        X, y = load_iris(return_X_y=True)
        k = 2
        sel = ConvexHullFeatureSelector(n_features_to_select=k, estimator=None)
        sel.fit(X, y)
        assert sel.transform(X).shape == (len(X), k)

    def test_selected_features_length(self):
        X, y = load_iris(return_X_y=True)
        k = 2
        sel = ConvexHullFeatureSelector(n_features_to_select=k, estimator=None)
        sel.fit(X, y)
        assert len(sel.selected_features_) == k

    def test_delta_ch_score_in_range(self):
        X, y = load_iris(return_X_y=True)
        sel = ConvexHullFeatureSelector(n_features_to_select=2, estimator=None)
        sel.fit(X, y)
        assert 0.0 <= sel.delta_ch_score_ <= 1.0

    def test_step2_runs_with_estimator(self):
        X, y = load_iris(return_X_y=True)
        sel = ConvexHullFeatureSelector(
            n_features_to_select=2,
            top_m=4,
            estimator=SVC(),
            cv=3,
        )
        sel.fit(X, y)
        assert sel.cv_score_ is not None
        assert 0.0 <= sel.cv_score_ <= 1.0

    def test_get_support_mask(self):
        X, y = load_iris(return_X_y=True)
        sel = ConvexHullFeatureSelector(n_features_to_select=2, estimator=None)
        sel.fit(X, y)
        mask = sel.get_support()
        assert mask.shape == (X.shape[1],)
        assert mask.sum() == 2

    def test_get_support_indices(self):
        X, y = load_iris(return_X_y=True)
        sel = ConvexHullFeatureSelector(n_features_to_select=2, estimator=None)
        sel.fit(X, y)
        idxs = sel.get_support(indices=True)
        assert len(idxs) == 2
        assert set(idxs) == set(sel.selected_features_)

    def test_get_feature_names_out(self):
        X, y = load_iris(return_X_y=True)
        names = load_iris().feature_names.tolist()
        sel = ConvexHullFeatureSelector(n_features_to_select=2, estimator=None)
        sel.fit(X, y)
        out = sel.get_feature_names_out(names)
        assert len(out) == 2
        for n in out:
            assert n in names

    def test_sklearn_pipeline_compatible(self):
        X, y = load_iris(return_X_y=True)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("sel",   ConvexHullFeatureSelector(
                n_features_to_select=2,
                top_m=3,
                estimator=SVC(),
                cv=3,
            )),
            ("clf",   SVC()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_invalid_k_raises(self):
        X, y = load_iris(return_X_y=True)
        sel = ConvexHullFeatureSelector(n_features_to_select=X.shape[1])
        with pytest.raises(ValueError, match="n_features_to_select"):
            sel.fit(X, y)

    def test_summary_is_string(self):
        X, y = load_iris(return_X_y=True)
        sel = ConvexHullFeatureSelector(n_features_to_select=2, estimator=None)
        sel.fit(X, y)
        assert isinstance(sel.summary(), str)

    def test_not_fitted_raises(self):
        from sklearn.exceptions import NotFittedError
        sel = ConvexHullFeatureSelector(n_features_to_select=2)
        with pytest.raises(NotFittedError):
            sel.transform(np.zeros((5, 4)))

    def test_fit_transform(self):
        X, y = load_iris(return_X_y=True)
        sel = ConvexHullFeatureSelector(n_features_to_select=2, estimator=None)
        X_out = sel.fit_transform(X, y)
        assert X_out.shape == (len(X), 2)

    def test_multiclass_iris(self):
        """Full two-step on 3-class Iris."""
        X, y = load_iris(return_X_y=True)
        sel = ConvexHullFeatureSelector(
            n_features_to_select=2,
            top_m=4,
            estimator=RandomForestClassifier(n_estimators=20, random_state=0),
            cv=3,
        )
        sel.fit(X, y)
        assert len(sel.selected_features_) == 2
        assert sel.cv_score_ > 0.5
