"""
Example 1 – Breast Cancer (Binary Classification)
==================================================
Reproduces the breast cancer experiment from Table 1 of the paper.

Dataset : Wisconsin Breast Cancer (30 features, 2 classes, 569 samples)
Task    : Select the best 3 features (k=3, d=30)
Step 2  : Random Forest classifier with 5-fold CV
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from convex_feature_select import ConvexHullFeatureSelector
from convex_feature_select.plot import plot_delta_ch_ranking, plot_feature_space_2d

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, 2 classes")

# ---------------------------------------------------------------------------
# 2. Run two-step feature selection
# ---------------------------------------------------------------------------
selector = ConvexHullFeatureSelector(
    n_features_to_select=3,
    top_m=50,
    estimator=RandomForestClassifier(n_estimators=100, random_state=0),
    cv=5,
    verbose=1,
)
selector.fit(X_train, y_train)
print()
print(selector.summary())

# ---------------------------------------------------------------------------
# 3. Evaluate on held-out test set
# ---------------------------------------------------------------------------
X_train_red = selector.transform(X_train)
X_test_red  = selector.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train_red, y_train)
acc = accuracy_score(y_test, clf.predict(X_test_red))
print(f"\nTest accuracy (RF on selected features): {acc:.4f}")

# ---------------------------------------------------------------------------
# 4. Visualise
# ---------------------------------------------------------------------------
# Bar chart of top-20 Δ_CH scores
fig1 = plot_delta_ch_ranking(
    selector.ranking_,
    top_n=20,
    highlight=selector.selected_features_,
    title="Breast Cancer – Top 20 feature subsets by Δ_CH",
    save_path="example1_ranking.png",
)

# 2-D feature space for the best two features in the selected subset
fi, fj = selector.selected_features_[0], selector.selected_features_[1]
fig2 = plot_feature_space_2d(
    X_train, y_train,
    feature_indices=(fi, fj),
    feature_names=feature_names,
    title=f"Feature space: {feature_names[fi]} vs {feature_names[fj]}",
    save_path="example1_feature_space.png",
)

print("Plots saved: example1_ranking.png, example1_feature_space.png")
