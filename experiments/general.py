import numpy as np
import matplotlib.pyplot as plt
from analysis.comparisons import compare_methods
from visualization.pca_dispersion import plot_pca_and_dispersion


def experiment_2(df_valid):
  print("=== GENERAL ANALYSIS (all genres) ===")
  print(df_valid['composer'].value_counts())
  X_orig_all = np.stack(df_valid['features_orig'].values)
  X_breg_all = np.stack(df_valid['features_breg'].values)
  composers = df_valid['composer'].values
  compare_methods(X_orig_all, X_breg_all, composers)

  fig = plt.figure(figsize=(14, 10))

  ax1 = fig.add_subplot(2, 2, 1)
  ax2 = fig.add_subplot(2, 2, 2)
  ax3 = fig.add_subplot(2, 2, 3)
  ax4 = fig.add_subplot(2, 2, 4)

  plot_pca_and_dispersion(X_orig_all, "Original method", ax1, ax3, composers)
  plot_pca_and_dispersion(X_breg_all, "Bregman-enhanced (α=1.0, β=1.0, λ=1.0)", ax2, ax4, composers)

  plt.tight_layout()
  plt.show()

  return {'X_orig_all': X_orig_all, 'X_breg_all': X_breg_all, 'composers': composers}
