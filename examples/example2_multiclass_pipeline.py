"""
Example 2 – Multi-class Classification & sklearn Pipeline
==========================================================
Demonstrates:
  * Multi-class support (Iris dataset, 3 classes)
  * Integration with a scikit-learn Pipeline
  * Comparing SVM and KNN via the same selector

The selector fits inside a Pipeline like any other transformer.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from convex_feature_select import ConvexHullFeatureSelector
from convex_feature_select.plot import plot_delta_ch_ranking, plot_feature_space_2d

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, 3 classes")
print(f"Selecting best 2 out of {X.shape[1]} features\n")

# ---------------------------------------------------------------------------
# 2. Use selector inside a Pipeline (with SVM as Step-2 estimator)
# ---------------------------------------------------------------------------
pipe = Pipeline([
    ("scaler",   StandardScaler()),
    ("selector", ConvexHullFeatureSelector(
        n_features_to_select=2,
        top_m=6,
        estimator=SVC(kernel="rbf"),
        cv=5,
        verbose=1,
    )),
    ("clf",      SVC(kernel="rbf")),
])

# fit the whole pipeline
pipe.fit(X, y)

selector = pipe.named_steps["selector"]
print()
print(selector.summary())

# ---------------------------------------------------------------------------
# 3. Cross-validate the full pipeline
# ---------------------------------------------------------------------------
scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"\n5-fold CV accuracy (full pipeline): {scores.mean():.4f} ± {scores.std():.4f}")

# ---------------------------------------------------------------------------
# 4. Compare SVM vs KNN on the selected subset
# ---------------------------------------------------------------------------
X_red = selector.transform(X)

for name, clf in [("SVM", SVC(kernel="rbf")), ("KNN", KNeighborsClassifier(n_neighbors=5))]:
    acc = cross_val_score(clf, X_red, y, cv=5).mean()
    print(f"  CV accuracy on selected features – {name}: {acc:.4f}")

# ---------------------------------------------------------------------------
# 5. Visualise
# ---------------------------------------------------------------------------
plot_delta_ch_ranking(
    selector.ranking_,
    highlight=selector.selected_features_,
    title="Iris – Δ_CH ranking (all 2-feature subsets)",
    save_path="example2_ranking.png",
)

plot_feature_space_2d(
    X, y,
    feature_indices=selector.selected_features_[:2],
    feature_names=feature_names,
    title="Iris – Selected feature space with convex hulls",
    save_path="example2_feature_space.png",
)

print("\nPlots saved: example2_ranking.png, example2_feature_space.png")
