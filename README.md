# convex-feature-select

A scikit-learn compatible Python package for **two-step, convex hull-based feature selection** for classification tasks.

---

## Background

This package implements the methodology proposed in:

> Du, D., Karve, P., & Mahadevan, S. (2025). *Feature selection for classification models using feature space geometric structure revealed by convex hulls.* (under review)

### Core idea

**Step 1 – Δ_CH screening (filter step)**

For each candidate feature subset *S*, construct one convex hull per class in the feature subspace.  A data point is *confusing* if it lies inside more than one hull — meaning the classes geometrically overlap at that point.  The dissimilarity metric is:

$$\Delta_{CH}(S) = \frac{1}{n_d} \sum_{P=1}^{n_d} \mathbf{1}\!\left[\sum_{i=1}^{n_c} HV_i^S(P) = 1\right]$$

Higher Δ_CH → better class separation → more promising feature subset.  All candidate subsets of size *k* are ranked, and the top-*m* are shortlisted.

**Step 2 – Exhaustive search over top-m subsets (wrapper step)**

Each shortlisted subset is evaluated by cross-validating a user-supplied classifier.  The best-scoring subset is selected.

**Result:** accuracy comparable to exhaustive wrapper search, at a fraction of the cost.

### Key properties of Δ_CH

- **Model-agnostic** — no classifier required in Step 1
- **Parameter-free** — convex hulls are deterministic given the data
- **Scale-invariant** — linear scaling does not change hull membership
- **Monotonic** — adding features can only increase Δ_CH, enabling efficient pruning

---

## Installation

```bash
pip install convex-hull-based-feature-select
```

Or from source:

```bash
git clone https://github.com/Dongjin-Du/convex-hull-based-feature-selection.git
cd convex-feature-select
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy, SciPy, scikit-learn, Matplotlib.

---

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from convex-hull-based-feature-select import ConvexHullFeatureSelector

X, y = load_breast_cancer(return_X_y=True)

selector = ConvexHullFeatureSelector(
    n_features_to_select=3,   # select k=3 features
    top_m=50,                  # evaluate top 50 subsets in Step 2
    estimator=RandomForestClassifier(n_estimators=100, random_state=0),
    cv=5,
    verbose=1,
)
selector.fit(X, y)
print(selector.summary())

X_reduced = selector.transform(X)  # shape (569, 3)
```

```
[ConvexHullFeatureSelector] Step 1: ranking subsets by Δ_CH ...
[ConvexHullFeatureSelector] Step 1 done.  Evaluated 4060 subsets.  Best Δ_CH = 0.9956  (50 subsets shortlisted for Step 2)
[ConvexHullFeatureSelector] Step 2: CV evaluation of 50 subsets ...
[ConvexHullFeatureSelector] Done.  Selected features: (...)

ConvexHullFeatureSelector – fit summary
  n_features_in          : 30
  n_features_to_select   : 3
  subsets evaluated      : 4060
  top_m shortlisted      : 50
  selected features      : (...)
  Δ_CH score             : ...
  CV accuracy            : ...
```

---

## API reference

### `ConvexHullFeatureSelector`

Scikit-learn compatible estimator (implements `fit`, `transform`, `fit_transform`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_features_to_select` | int | 3 | Cardinality of selected subset |
| `top_m` | int | 50 | Top-m subsets shortlisted for Step 2 |
| `estimator` | sklearn classifier | None | Used in Step 2; if None, skips Step 2 |
| `cv` | int | 5 | Cross-validation folds |
| `scoring` | str | `'accuracy'` | CV scoring metric |
| `max_subsets` | int | None | Cap on Step 1 search space |
| `n_jobs` | int | 1 | Parallel jobs for CV |
| `verbose` | int | 0 | Verbosity (0/1/2) |

**Fitted attributes:**

| Attribute | Description |
|---|---|
| `selected_features_` | Tuple of selected feature indices |
| `delta_ch_score_` | Δ_CH value of the selected subset |
| `cv_score_` | Mean CV accuracy (None if Step 2 skipped) |
| `ranking_` | Full Step 1 ranking `[(indices, delta_ch), …]` |
| `n_features_in_` | Number of input features |

**Methods:** `fit`, `transform`, `fit_transform`, `get_support`, `get_feature_names_out`, `summary`

---

### `delta_ch(X, y, feature_indices=None)`

Compute Δ_CH for a single feature subset.

```python
from convex_feature_select import delta_ch
score = delta_ch(X, y, feature_indices=[2, 7, 21])
```

---

### `rank_subsets(X, y, n_features_to_select, *, max_subsets=None)`

Rank all C(n_features, k) subsets by Δ_CH.

```python
from convex_feature_select import rank_subsets
ranking = rank_subsets(X, y, n_features_to_select=3)
# [(feature_tuple, delta_ch_score), …] sorted descending
```

---

## Using inside a scikit-learn Pipeline

The selector integrates seamlessly with sklearn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from convex_feature_select import ConvexHullFeatureSelector

pipe = Pipeline([
    ("scaler",   StandardScaler()),
    ("selector", ConvexHullFeatureSelector(
        n_features_to_select=3,
        top_m=50,
        estimator=SVC(),
        cv=5,
    )),
    ("clf",      SVC()),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```

---

## Examples

| File | Description |
|---|---|
| `examples/example1_breast_cancer.py` | Binary classification, breast cancer dataset |
| `examples/example2_multiclass_pipeline.py` | Multi-class Iris + sklearn Pipeline |
| `examples/example3_metric_only.py` | Low-level `delta_ch` and `rank_subsets` API |

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Computational complexity

| Step | Complexity |
|---|---|
| Convex hull construction (Quickhull) | O(n log n) average |
| Point-in-hull query (Delaunay) | O(n log n) average |
| **Δ_CH per subset** | **O(n log n)** |
| Step 1 (all subsets) | O(C(d,k) · n log n) |
| Step 2 (top-m CV) | O(m · cv · model_fit_cost) |

The monotonicity of Δ_CH means only subsets of size exactly *k* need to be evaluated — subsets of smaller size can be safely pruned.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you use this package in academic work, please cite:

```bibtex
@article{du2025convex,
  title   = {Feature selection for classification models using feature space
             geometric structure revealed by convex hulls},
  author  = {Du, Dongjin and Karve, Pranav and Mahadevan, Sankaran},
  journal = {(under review)},
  year    = {2025},
}
```
