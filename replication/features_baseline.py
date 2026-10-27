"""Hand-crafted-feature baselines and spectral characterization (no deep learning).

Features per normalized window (DC removed, unit RMS): 64-bin log power spectrum (FFT of length 256,
averaged in groups of 4 bins, centered), amplitude statistics (std, kurtosis and peak-to-average power of
|x|), and fourth-order cumulant magnitudes |C40|, |C42| (classical AMC features). Models: multinomial
logistic regression on standardized features and a histogram gradient-boosting classifier.
Reported for synthetic Dataset v5 (train -> test) and for each hardware LODO fold (two transmitters'
train split -> held-out transmitter).
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

from run_experiment import normalize, syn_data, real_data
from config import CACHE, DATA, FIGURES, HERE, RESULTS, RUNS

OUT = RESULTS


def feats(x):
    z = x[..., 0] + 1j * x[..., 1]
    P = np.abs(np.fft.fftshift(np.fft.fft(z, axis=1), axes=1)) ** 2 / z.shape[1]
    lp = np.log10(P.reshape(len(z), 64, 4).mean(2) + 1e-6)
    a = np.abs(z)
    am = a.mean(1, keepdims=True)
    kurt = ((a - am) ** 4).mean(1) / (a.var(1) ** 2 + 1e-12)
    papr = (a ** 2).max(1) / ((a ** 2).mean(1) + 1e-12)
    c20 = (z ** 2).mean(1)
    c21 = (np.abs(z) ** 2).mean(1)
    c40 = (z ** 4).mean(1) - 3 * c20 ** 2
    c42 = (np.abs(z) ** 4).mean(1) - np.abs(c20) ** 2 - 2 * c21 ** 2
    return np.column_stack([lp, a.std(1), kurt, np.log10(papr), np.abs(c40), np.abs(c42), np.abs(c20)])


def evaluate(y, p):
    pred = (p >= 0.5).astype(int)
    rf = float(np.mean(pred[y == 0] == 0))
    ra = float(np.mean(pred[y == 1] == 1))
    return {"acc": float(np.mean(pred == y)), "rec_F": rf, "rec_A": ra, "bacc": 0.5 * (rf + ra)}


def fit_eval(Ftr, ytr, Fte, yte, wts, extra=None):
    """Fit both baselines; extra maps a subset name to a boolean mask over the test windows."""
    out = {}
    sw = np.where(ytr == 1, wts[1], wts[0])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    lr.fit(Ftr, ytr, logisticregression__sample_weight=sw)
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
    gb.fit(Ftr, ytr, sample_weight=sw)
    for name, mdl in [("logreg", lr), ("gboost", gb)]:
        p = mdl.predict_proba(Fte)[:, 1]
        out[name] = evaluate(yte, p)
        for k, msk in (extra or {}).items():
            out[name][k] = evaluate(yte[msk], p[msk])
    return out


def main():
    res = {}
    X, m = syn_data()
    occ = m.y_state.values > 0
    tr, te = occ & (m.split == "train").values, occ & (m.split == "test").values
    Ftr, Fte = feats(normalize(X[tr])), feats(normalize(X[te]))
    ytr, yte = m.y_state.values[tr] - 1, m.y_state.values[te] - 1
    w = len(ytr) / (2 * np.bincount(ytr)); w = w / w.mean()
    snr = m.snr_db.values[te]
    res["synthetic"] = fit_eval(Ftr, ytr, Fte, yte, w, {"informative": snr >= -10})
    print("synthetic:", json.dumps(res["synthetic"]))
    R, rm = real_data()
    yr = (rm.cls == "Adversarial").astype(int).values
    Fr = feats(normalize(np.asarray(R)))
    for held in ["Pluto_02", "Pluto_04", "Pluto_05"]:
        trd = (rm.tx != held).values & (rm.split == "train").values
        ted = (rm.tx == held).values
        w = len(yr[trd]) / (2 * np.bincount(yr[trd])); w = w / w.mean()
        mods = rm.modulation.values[ted]
        res[f"hw_{held}"] = fit_eval(Fr[trd], yr[trd], Fr[ted], yr[ted], w,
                                     {"PAM4": mods == "PAM4", "PAM8": mods == "PAM8"})
        print(held, json.dumps(res[f"hw_{held}"]))
    json.dump(res, open(f"{OUT}/features_baseline.json", "w"), indent=1)


if __name__ == "__main__":
    main()
