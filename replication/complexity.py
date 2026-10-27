"""Parameter count and multiply-accumulate operations (MACs) per 256-sample window for each architecture."""
import json, sys
import numpy as np
import npdl as nd
import models
from config import CACHE, DATA, FIGURES, HERE, RESULTS, RUNS


def macs(model, L=256):
    total, T, F = 0, L, 2
    for l in model.layers:
        total, T, F = step(l, total, T, F)
    return total


def step(l, total, T, F):
    if isinstance(l, nd.Conv1D):
        return total + T * l.k * l.cin * l.cout, T, l.cout
    if isinstance(l, nd.MaxPool1D):
        return total, T // 2, F
    if isinstance(l, nd.Patchify):
        return total, T // l.p, F * l.p
    if isinstance(l, nd.LSTM):
        h = l.h
        return total + T * 4 * h * (F + h), (T if l.return_seq else 1), h
    if isinstance(l, nd.GRU):
        h = l.h
        return total + T * 3 * h * (F + h), (T if l.return_seq else 1), h
    if isinstance(l, nd.BiLSTM):
        h = l.h
        return total + 2 * T * 4 * h * (F + h), 1, 2 * h
    if isinstance(l, nd.Dense):
        nin, nout = l.W.v.shape
        return total + T * nin * nout, T, nout
    if isinstance(l, nd.GlobalAvgPool):
        return total, 1, F
    if isinstance(l, nd.Residual):
        for s in l.layers:
            if isinstance(s, nd.MHSA):
                d = s.d
                total += 4 * T * d * d + 2 * T * T * d
            else:
                total, T, F = step(s, total, T, F)
        return total, T, F
    return total, T, F


out = {}
for a in ["cnn", "cnn_lstm", "cnn_gru", "cldnn", "bilstm", "transformer"]:
    m = models.build(a, 2, 0)
    out[a] = {"params": m.n_params(), "macs": int(macs(m))}
    print(f"{models.ARCH_NAMES[a]:22s} params {out[a]['params']:8,d}  MACs/window {out[a]['macs'] / 1e6:6.2f} M")
json.dump(out, open(f"{RESULTS}/complexity.json", "w"), indent=1)
