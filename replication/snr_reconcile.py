"""Reviewer 2, C1: window SNR measured on the in-phase axis only versus on the complex residual."""
import json, sys
import numpy as np, pandas as pd
import hw_diagnostics as H
from run_experiment import real_data
from config import RESULTS
R, rm = real_data(); n = np.arange(256); rows = []
rng = np.random.default_rng(0)
for mod in ("BPSK", "PAM4", "PAM8"):
    for tx in ("Pluto_02", "Pluto_04", "Pluto_05"):
        idx = np.where((rm.modulation == mod).values & (rm.tx == tx).values)[0]
        for i in rng.choice(idx, size=min(150, len(idx)), replace=False):
            z = R[i, :, 0].astype(np.float64) + 1j * R[i, :, 1]; z = z - z.mean()
            f_exp = H.alias(H.SUB[mod] + H.CFO[tx])
            S = np.abs(np.fft.fft(z ** 2, 4096)); fr = np.fft.fftfreq(4096, 1 / H.FS); line = fr[S.argmax()]
            f = min([(line + k * H.FS) / 2 for k in range(-3, 4)], key=lambda c: abs(H.alias(c - f_exp)))
            zd = z * np.exp(-2j * np.pi * f * n / H.FS); phi = 0.5 * np.angle((zd ** 2).sum()); w = zd * np.exp(-1j * phi)
            y, q = w.real, w.imag
            lv, resid = H.kmeans1d(y, {"PAM4": 4, "PAM8": 8, "BPSK": 2}[mod])
            sig = np.mean((y - resid) ** 2)
            rows.append(dict(mod=mod, tx=tx, inphase_db=10 * np.log10(np.mean(y ** 2) / np.mean(resid ** 2)),
                             complex_db=10 * np.log10(sig / (np.mean(resid ** 2) + np.mean(q ** 2)))))
d = pd.DataFrame(rows); g = d.groupby(["mod", "tx"]).median().round(2)
print(g); out = {"by_mod_tx": {f"{a}|{b}": r for (a, b), r in g.to_dict(orient="index").items()},
                 "complex_range_db": [float(g.complex_db.min()), float(g.complex_db.max())],
                 "inphase_range_db": [float(g.inphase_db.min()), float(g.inphase_db.max())]}
json.dump(out, open(f"{RESULTS}/paper/snr_reconcile.json", "w"), indent=1); print(out["complex_range_db"], out["inphase_range_db"])
