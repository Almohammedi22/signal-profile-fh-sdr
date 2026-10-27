"""Test-time frequency-shift intervention on trained LODO models (no retraining).

Each held-out window is rotated by exp(j 2 pi df n / fs) after preprocessing, for df on a grid.
Shifting the held-out transmitter's offset into or out of the range spanned by the training
transmitters tests whether the PAM4/PAM8 errors follow the carrier-frequency offset.
Also reports the 'calibrated' shift that moves the held-out offset to the mean training offset.
Usage: python cfo_shift_eval.py   (processes every finished 2-class hw run without a cfo_shift.json)
"""
import glob, json, os, sys
import numpy as np, pandas as pd

import npdl as nd
import models
from run_experiment import normalize, rotate, real_data
from config import CACHE, DATA, FIGURES, HERE, RESULTS, RUNS

CFO = {"Pluto_02": -1387.4, "Pluto_04": -658.0, "Pluto_05": -2403.7}     # medians from cfo_offsets.py
GRID = np.arange(-3000, 3001, 500.0)
MODS = ["PAM4", "PAM8", "ASK", "OOK", "BPSK", "QPSK", "16QAM", "64QAM"]


def load_model(run_dir, arch, ncls, seed):
    m = models.build(arch, ncls, seed)
    w = np.load(f"{run_dir}/weights.npz")
    arrs = [w[f"arr_{i}"] for i in range(len(w.files))]
    ps = m.params()
    for p, a in zip(ps, arrs[:len(ps)]):
        p.v[...] = a
    rest = arrs[len(ps):]
    for b, (rm_, rv_) in zip(m.bn_layers(), zip(rest[0::2], rest[1::2])):
        b.rm[...] = rm_
        b.rv[...] = rv_
    return m


def main():
    R, rm = real_data()
    for f in sorted(glob.glob(f"{RUNS}/hw_*/info.json")):
        d = os.path.dirname(f)
        info = json.load(open(f))
        if info["ncls"] != 2 or os.path.exists(f"{d}/cfo_shift.json"):
            continue
        held = "_".join(info["job"].split("_")[1:3])
        train_devs = [t for t in CFO if t != held]
        calib = float(np.mean([CFO[t] for t in train_devs]) - CFO[held])
        sel = (rm.tx == held).values & rm.modulation.isin(MODS).values
        X = normalize(R[sel])
        mods = rm.modulation.values[sel]
        y = (rm.cls.values[sel] == "Adversarial").astype(int)
        model = load_model(d, info["arch"], 2, info["seed"])
        out = {"heldout": held, "calibrated_shift_hz": calib, "grid_hz": GRID.tolist(), "recall": {}}
        for df in list(GRID) + [calib]:
            p = nd.softmax(models.predict_logits(model, rotate(X, df)))[:, 1]
            key = "calibrated" if df == calib else f"{df:+.0f}"
            out["recall"][key] = {mm: float(np.mean((p[mods == mm] >= 0.5) == (y[mods == mm] == 1))) for mm in MODS}
        json.dump(out, open(f"{d}/cfo_shift.json", "w"), indent=1)
        r0, rc = out["recall"]["+0"], out["recall"]["calibrated"]
        print(f"{info['job']}: PAM8 {r0['PAM8']:.3f} -> {rc['PAM8']:.3f}, PAM4 {r0['PAM4']:.3f} -> {rc['PAM4']:.3f} "
              f"(calibrated shift {calib:+.0f} Hz)", flush=True)


if __name__ == "__main__":
    main()
