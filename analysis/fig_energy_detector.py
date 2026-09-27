"""Regenerate the energy-detector figure (three panels, one PDF each) from the project data.

(a) ROC of the energy detector on the Dataset v5 test split, (b) detection probability versus SNR at
the saved threshold, (c) accuracy and balanced accuracy of the complete three-state pipeline versus
the threshold. Usage: python fig_energy_detector.py [data_folder]
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy.io as sio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mat73 import Mat73  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "."
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8})

# data: test split of Dataset v5, saved Stage 1 threshold, saved classifier predictions
m = Mat73(os.path.join(root, "dataset_v5.mat"))
test = m.read("test_idx").ravel().astype(int) - 1
yt = m.read("y_state").ravel().astype(int)[test]
st = m.read("snr_db").ravel()[test]
X = m.read("X")[test].astype(np.float64)
Et = np.mean(np.sum(X ** 2, axis=2), axis=1)
thr = float(sio.loadmat(os.path.join(root, "energy_gate_v5.mat"), squeeze_me=True)["threshold"])
pr = pd.read_csv(os.path.join(root, "cnn_lstm_occ_v5_snr_binned_predictions.csv"))
pos = {int(t): i for i, t in enumerate(test)}
ti = np.array([pos[int(r) - 1] for r in pr.OriginalDatasetIndex])
cls = np.full(len(yt), -1)
cls[ti] = pr.PredictedLabel.map({"friendly": 1, "adversarial": 2}).values
occ = yt > 0

# (a) ROC
ts = np.unique(np.concatenate([np.quantile(Et, np.linspace(0, 1, 4001)), [thr]]))
pd_ = np.array([np.mean(Et[occ] > t) for t in ts])
pfa = np.array([np.mean(Et[~occ] > t) for t in ts])
auc = abs(np.trapezoid(pd_, pfa))
op = (np.mean(Et[~occ] > thr), np.mean(Et[occ] > thr))


def pipeline(t):
    """Accuracy and three-state balanced accuracy of the complete pipeline at threshold t."""
    final = np.where(Et <= t, 0, np.where(yt == 0, 9, cls))   # 9 = idle observation passed to Stage 2
    return np.mean(final == yt), np.mean([np.mean(final[yt == c] == c) for c in (0, 1, 2)])


grid = np.unique(np.concatenate([np.geomspace(0.85, 3.0, 600), [thr]]))
res = np.array([pipeline(t) for t in grid])
acc, bal = pipeline(thr)
print(f"AUC {auc:.3f} | Pfa {op[0] * 100:.1f}% | Pd {op[1] * 100:.1f}% | pipeline {acc * 100:.2f}% "
      f"(balanced {bal * 100:.2f}%) | best over thresholds {res[:, 0].max() * 100:.2f}%")

fig, a = plt.subplots(figsize=(2.25, 2.0))
a.plot(pfa, pd_, color="#1f4e79", lw=1.1, label=f"energy detector (AUC {auc:.3f})")
a.plot([0, 1], [0, 1], color="#999999", lw=0.6, ls=":")
a.plot(*op, "o", ms=4.5, mfc="white", mec="#b03a2e", mew=1.1, label="selected threshold")
a.set(xlabel="False-alarm probability", ylabel="Detection probability", xlim=(0, 1), ylim=(0, 1.01))
a.legend(loc="lower right", fontsize=6.6, frameon=False)
fig.savefig("fig_energy_a_roc.pdf", bbox_inches="tight")

# (b) detection probability versus SNR
fig, a = plt.subplots(figsize=(2.25, 2.0))
edges = np.arange(-20, 25.1, 2.5)
for c, name, col, mk in [(1, "friendly profiles", "#1f4e79", "o"), (2, "adversarial profiles", "#b03a2e", "s")]:
    pdc = [np.mean(Et[(yt == c) & (st >= lo) & (st < lo + 2.5)] > thr) for lo in edges[:-1]]
    a.plot(edges[:-1] + 1.25, pdc, marker=mk, ms=2.8, lw=0.9, color=col, label=name)
a.axhline(op[0], color="#777777", lw=0.7, ls="--")
a.text(24, op[0] + 0.03, "false-alarm rate", ha="right", fontsize=6.4, color="#555555")
a.set(xlabel="SNR (dB)", ylabel="Detection probability", ylim=(0, 1.03), xlim=(-20, 25))
a.legend(loc="center right", fontsize=6.6, frameon=False)
fig.savefig("fig_energy_b_detection_snr.pdf", bbox_inches="tight")

# (c) complete pipeline versus threshold
fig, a = plt.subplots(figsize=(2.25, 2.0))
a.plot(grid, 100 * res[:, 0], color="#1f4e79", lw=1.1, label="accuracy")
a.plot(grid, 100 * res[:, 1], color="#1f4e79", lw=0.9, ls="--", label="balanced accuracy")
a.axvline(thr, color="#b03a2e", lw=0.8)
a.text(thr * 1.04, 88, f"selected {thr:.3f}", fontsize=6.4, color="#b03a2e")
a.set_xscale("log")
a.set_xticks([0.9, 1.2, 1.5, 2, 3])
a.set_xticklabels(["0.9", "1.2", "1.5", "2", "3"])
a.minorticks_off()
a.set(xlabel="Energy threshold (normalized)", ylabel="Three-state result (%)", ylim=(0, 100))
a.legend(loc="lower right", fontsize=6.6, frameon=False)
fig.savefig("fig_energy_c_pipeline.pdf", bbox_inches="tight")
