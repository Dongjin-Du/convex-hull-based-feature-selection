"""
Example 3 – Using Δ_CH Directly (No Step 2)
=============================================
Sometimes you just want to use the Δ_CH metric on its own—to rank subsets
or to score a specific subset without running cross-validation.

This example shows the low-level API: ``delta_ch`` and ``rank_subsets``.
"""

import numpy as np
from convex_feature_select import delta_ch, rank_subsets

# ---------------------------------------------------------------------------
# Synthetic dataset: 2 informative features + 2 noise features
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
n   = 200

X = np.hstack([
    np.vstack([rng.normal([0, 0], 0.4, (n//2, 2)),
               rng.normal([3, 3], 0.4, (n//2, 2))]),   # informative
    rng.normal(0, 1, (n, 2)),                            # noise
])
y = np.array([0]*(n//2) + [1]*(n//2))

print("Feature indices: 0,1 = informative | 2,3 = noise\n")

# ---------------------------------------------------------------------------
# Score specific subsets
# ---------------------------------------------------------------------------
for subset in [(0, 1), (2, 3), (0, 2), (1, 3), (0, 1, 2, 3)]:
    score = delta_ch(X, y, subset)
    print(f"  Δ_CH{subset} = {score:.4f}")

# ---------------------------------------------------------------------------
# Rank all 2-feature subsets
# ---------------------------------------------------------------------------
print("\nRanking all 2-feature subsets:")
ranking = rank_subsets(X, y, n_features_to_select=2)
for rank, (subset, score) in enumerate(ranking, 1):
    print(f"  #{rank}  features={subset}  Δ_CH={score:.4f}")
