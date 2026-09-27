"""End-to-end evaluation of the two-stage pipeline on the Dataset v5 test split (no retraining).

Stage 1: energy detector with the saved threshold (energy_gate_v5.mat). Stage 2: the saved CNN-LSTM
predictions (cnn_lstm_occ_v5_snr_binned_predictions.csv), joined by OriginalDatasetIndex.
The recomputed Stage 1 confusion must match the saved one exactly, or the script stops.
Usage: python pipeline_end_to_end.py [project_folder]
"""
import sys, os
import numpy as np, pandas as pd, scipy.io as sio
from mat73 import Mat73

root = sys.argv[1] if len(sys.argv) > 1 else "."
m = Mat73(os.path.join(root, "dataset_v5.mat"))
y_all, snr_all = m.read("y_state").ravel().astype(int), m.read("snr_db").ravel()
test = m.read("test_idx").ravel().astype(int) - 1                 # MATLAB 1-based -> 0-based
X = m.read("X")                                                    # N x 256 x 2
E = np.mean(np.sum(X[test].astype(np.float64) ** 2, axis=2), axis=1)
gate = sio.loadmat(os.path.join(root, "energy_gate_v5.mat"), squeeze_me=True)
thr, saved_cm = float(gate["threshold"]), np.asarray(gate["gateConfusionMatrix"])
y, snr, det = y_all[test], snr_all[test], E > thr
cm = np.array([[np.sum((y == 0) & ~det), np.sum((y == 0) & det)], [np.sum((y > 0) & ~det), np.sum((y > 0) & det)]])
assert (cm == saved_cm).all(), f"Stage 1 reproduction failed: {cm.tolist()} vs {saved_cm.tolist()}"

pr = pd.read_csv(os.path.join(root, "cnn_lstm_occ_v5_snr_binned_predictions.csv"))
pos = {int(t): i for i, t in enumerate(test)}
ti = np.array([pos[int(r) - 1] for r in pr.OriginalDatasetIndex])
assert np.allclose(snr[ti], pr.SNR_dB.values), "prediction file does not match the dataset"
lab = {"friendly": 1, "adversarial": 2}
cls = np.full(len(test), -1); cls[ti] = pr.PredictedLabel.map(lab).values
final = np.where(~det, 0, np.where(y == 0, 9, cls))              # 9 = idle window passed to Stage 2
names = {0: "idle", 1: "friendly", 2: "adversarial", 9: "false alarm to Stage 2"}
print("Stage 1 reproduced exactly:", cm.tolist())
print(pd.crosstab(pd.Series(y, name="true").map(names), pd.Series(final, name="pipeline").map(names)))
print(f"end-to-end accuracy: {np.sum(final == y)}/{len(y)} = {100 * np.mean(final == y):.2f}%")
rows = []
for lo, hi in [(-20, -10), (-10, 0), (0, 10), (10, 25.01)]:
    for c, n in [(1, "friendly"), (2, "adversarial")]:
        k = (y == c) & (snr >= lo) & (snr < hi)
        rows.append({"snr_bin": f"[{lo},{int(hi)})", "class": n, "n": int(k.sum()),
                     "missed_by_stage1_pct": round(100 * np.mean(final[k] == 0), 1),
                     "correct_pct": round(100 * np.mean(final[k] == c), 1),
                     "misclassified_pct": round(100 * np.mean((final[k] != c) & (final[k] != 0)), 1)})
out = pd.DataFrame(rows); out.to_csv("pipeline_end_to_end_by_snr.csv", index=False); print(out.to_string(index=False))
