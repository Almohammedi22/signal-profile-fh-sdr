"""Summarize cfo_shift_wide.json: recall inside and outside the effective offsets covered by the +-3 kHz augmentation."""
import glob, json
from config import RUNS, RESULTS
CFO = {"Pluto_02": -1387.4, "Pluto_04": -658.0, "Pluto_05": -2403.7}
out = {}
for f in sorted(glob.glob(f"{RUNS}/hw_*cfo3k/cfo_shift_wide.json")):
    j = json.load(open(f)); held = j["heldout"]; tr = [CFO[t] for t in CFO if t != held]
    lo, hi = min(tr) - 3000, max(tr) + 3000; rows = []
    for k, r in j["recall"].items():
        e = CFO[held] + float(k); rows.append((max(lo - e, e - hi, 0.0), e, r))
    inside = [r for d, e, r in rows if d == 0]
    out[j["job"]] = {"covered_khz": [lo / 1e3, hi / 1e3],
                     "inside_min_recall": {m: min(r[m] for r in inside) for m in inside[0]},
                     "outside": [{"distance_khz": d / 1e3, "effective_khz": e / 1e3, "recall": r} for d, e, r in sorted(rows, key=lambda x: x[0]) if d > 0]}
json.dump(out, open(f"{RESULTS}/paper/shift_wide_summary.json", "w"), indent=1)
for k, v in out.items(): print(k, {m: round(x, 3) for m, x in v["inside_min_recall"].items()})
