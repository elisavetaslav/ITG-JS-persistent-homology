import numpy as np
import matplotlib.pyplot as plt


def plot_barcodes(dgms, title="Persistence Barcodes", colors=None):
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e']  # default: blue for H0, orange for H1

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    for dim, ax in enumerate(axes):
        dgm = np.asarray(dgms[dim], dtype=float).copy()
        finite_mask = np.isfinite(dgm[:, 1])
        dgm = dgm[finite_mask]

        if len(dgm) == 0:
            ax.text(0.5, 0.5, f"No finite bars in H{dim}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"H{dim} barcode (no finite bars)")
            continue

        for i, (birth, death) in enumerate(dgm):
            ax.hlines(y=i, xmin=birth, xmax=death,
                      color=colors[dim], linewidth=2.5)

        ax.set_xlabel("Filtration value")
        ax.set_ylabel("Barcode index")
        ax.set_title(f"H{dim} barcode")
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)
        ax.invert_yaxis()
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
