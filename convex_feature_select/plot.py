"""
convex_feature_select.plot
==========================
Visualization helpers for the convex hull feature selector.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull, QhullError

from .metric import hull_vectors


__all__ = ["plot_delta_ch_ranking", "plot_feature_space_2d"]


def plot_delta_ch_ranking(
    ranking: list[tuple[tuple[int, ...], float]],
    *,
    top_n: int = 20,
    highlight: tuple[int, ...] | None = None,
    title: str = "Δ_CH ranking of feature subsets",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of Δ_CH scores for the top-n subsets.

    Parameters
    ----------
    ranking : list of (feature_tuple, delta_ch)
        Output of :func:`~convex_feature_select.rank_subsets`.
    top_n : int
        Number of subsets to display.
    highlight : tuple of int, optional
        Highlight the bar corresponding to this feature subset in red.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    top = ranking[:top_n]
    labels  = [str(feat) for feat, _ in top]
    scores  = [score for _, score in top]
    colors  = ["#C44E52" if (highlight is not None and feat == highlight)
               else "#4C72B0"
               for feat, _ in top]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, top_n * 0.55), 4))
    else:
        fig = ax.get_figure()

    bars = ax.bar(range(len(top)), scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Δ_CH", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.05)

    # Annotate bars with values
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center", va="bottom", fontsize=7,
        )

    if highlight is not None:
        patch = mpatches.Patch(color="#C44E52", label=f"Selected: {highlight}")
        ax.legend(handles=[patch], fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_space_2d(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: tuple[int, int],
    *,
    feature_names: Sequence[str] | None = None,
    title: str = "Feature space with convex hulls",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of two features with convex hulls overlaid.

    Confusing points (inside >1 hull) are marked differently from
    non-confusing ones, giving visual intuition for Δ_CH.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    feature_indices : (int, int)
        Pair of feature indices to plot.
    feature_names : sequence of str, optional
        Names for all features (for axis labels).
    title : str
    ax : matplotlib Axes, optional
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    i, j  = feature_indices
    X_sub = X_arr[:, [i, j]]

    classes   = np.unique(y_arr)
    cmap      = plt.cm.tab10(np.linspace(0, 0.85, len(classes)))
    class_col = {cls: cmap[k] for k, cls in enumerate(classes)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    # Compute hull vectors to identify confusing points
    hv = hull_vectors(X_sub, y_arr)
    confusing = hv.sum(axis=1) > 1

    # --- Draw convex hulls --------------------------------------------------
    for cls in classes:
        pts = X_sub[y_arr == cls]
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])
            ax.fill(
                pts[verts, 0], pts[verts, 1],
                alpha=0.08, color=class_col[cls],
            )
            ax.plot(
                pts[verts, 0], pts[verts, 1],
                color=class_col[cls], linewidth=1.5, linestyle="--",
            )
        except QhullError:
            pass

    # --- Scatter points -----------------------------------------------------
    legend_handles = []
    for cls in classes:
        mask = (y_arr == cls) & ~confusing
        ax.scatter(
            X_sub[mask, 0], X_sub[mask, 1],
            c=[class_col[cls]], s=30, edgecolors="white",
            linewidths=0.4, label=f"Class {cls}",
            zorder=3,
        )
        legend_handles.append(
            mpatches.Patch(color=class_col[cls], label=f"Class {cls}")
        )

    # Confusing points in a distinct marker
    if confusing.any():
        ax.scatter(
            X_sub[confusing, 0], X_sub[confusing, 1],
            c="black", s=35, marker="x", linewidths=1.2,
            label="Confusing", zorder=4,
        )
        legend_handles.append(
            mpatches.Patch(color="black", label="Confusing points")
        )

    # --- Labels / decorations -----------------------------------------------
    if feature_names is not None:
        ax.set_xlabel(feature_names[i], fontsize=12)
        ax.set_ylabel(feature_names[j], fontsize=12)
    else:
        ax.set_xlabel(f"Feature {i}", fontsize=12)
        ax.set_ylabel(f"Feature {j}", fontsize=12)

    delta = hv.sum(axis=1).clip(0, 1)       # count as 1 if >=1 hull
    non_conf_frac = (hv.sum(axis=1) == 1).mean()
    ax.set_title(f"{title}\nΔ_CH = {non_conf_frac:.3f}", fontsize=13)
    ax.legend(handles=legend_handles, fontsize=9, loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
