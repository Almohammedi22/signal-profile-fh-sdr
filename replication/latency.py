"""Inference time per window of the six NumPy networks on one CPU thread (run with OPENBLAS_NUM_THREADS=1).

Reports the median over repeats of the time per window at batch size 256, and the time for a single window.
Writes {RESULTS}/latency.json ({arch: ms per window at batch 256}) and latency_detail.json.
"""
import json, os, sys, time
import numpy as np

import models
from config import CACHE, DATA, FIGURES, HERE, RESULTS, RUNS

ARCHS = ["cnn", "cnn_lstm", "cnn_gru", "cldnn", "bilstm", "transformer"]


def timeit(m, x, reps):
    ts = []
    for _ in range(reps):
        t = time.process_time()
        models.predict_logits(m, x, bs=len(x))
        ts.append(time.process_time() - t)
    return float(np.median(ts))


def main():
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((256, 256, 2)).astype(np.float32)
    x1 = xb[:1]
    out, det = {}, {}
    for a in ARCHS:
        m = models.build(a, 2, 1)
        models.predict_logits(m, xb, bs=256)
        tb = timeit(m, xb, 7) / 256 * 1e3
        t1 = timeit(m, x1, 31) * 1e3
        out[a] = round(tb, 3)
        det[a] = {"ms_per_window_batch256": round(tb, 3), "ms_single_window": round(t1, 3)}
        print(a, det[a], flush=True)
    det["conditions"] = {"threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"), "numpy": np.__version__,
                         "cpu": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": ")}
    json.dump(out, open(f"{RESULTS}/latency.json", "w"), indent=1)
    json.dump(det, open(f"{RESULTS}/latency_detail.json", "w"), indent=1)


if __name__ == "__main__":
    main()
