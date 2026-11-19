"""Round-2 extension of the frequency-shift test (Reviewer 2, B1 and C3).
Same procedure as cfo_shift_eval.py, on a wider grid (-10 to +10 kHz, 1 kHz steps; the first run,
hw_Pluto_02_real_s1, was computed on a 500 Hz grid before the step was changed),
for every real-only and offset-augmented hardware network. Output: cfo_shift_wide.json per run."""
import glob, json, os, sys
import numpy as np
import npdl as nd, models
from run_experiment import normalize, rotate, real_data
from cfo_shift_eval import load_model, MODS
from config import RUNS
GRID = np.arange(-10000, 10001, 1000.0)
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else None
R, rm = real_data()
for f in sorted(glob.glob(f"{RUNS}/hw_*/info.json")):
    d = os.path.dirname(f); info = json.load(open(f))
    if (ONLY and info["job"] not in ONLY) or info["ncls"] != 2 or "combined" in info["job"] or os.path.exists(f"{d}/cfo_shift_wide.json"):
        continue
    held = "_".join(info["job"].split("_")[1:3])
    sel = (rm.tx == held).values & rm.modulation.isin(MODS).values
    X = normalize(R[sel]); mods = rm.modulation.values[sel]; y = (rm.cls.values[sel] == "Adversarial").astype(int)
    model = load_model(d, info["arch"], 2, info["seed"])
    out = {"job": info["job"], "heldout": held, "grid_hz": GRID.tolist(), "recall": {}}
    for df in GRID:
        p = nd.softmax(models.predict_logits(model, rotate(X, df)))[:, 1]
        out["recall"][f"{df:+.0f}"] = {m: float(np.mean((p[mods == m] >= 0.5) == (y[mods == m] == 1))) for m in MODS}
    json.dump(out, open(f"{d}/cfo_shift_wide.json", "w"), indent=1)
    r = out["recall"]; print(info["job"], "PAM8 at -10/-5/-3/0/+3/+5/+10 kHz:", [round(r[k]["PAM8"], 2) for k in ("-10000", "-5000", "-3000", "+0", "+3000", "+5000", "+10000")], flush=True)
