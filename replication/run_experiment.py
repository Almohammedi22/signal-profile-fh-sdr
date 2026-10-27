"""One training run of the replication study.

  python run_experiment.py syn <arch> <seed> [ncls]               synthetic Dataset v5 (occupied subset)
  python run_experiment.py hw <heldout> <regime> <seed> [ncls] [cfo_khz]
        heldout in {Pluto_02, Pluto_04, Pluto_05}; regime in {real, combined}
        cfo_khz > 0 adds random frequency rotations in [-cfo_khz, +cfo_khz] kHz to real training windows

Every input window is DC-removed and RMS-normalized per capture (Section 4.2). Results go to
{RUNS}/<job>/: info.json (configuration, training history, counts), scores.npz
(per-window class probabilities and labels for every evaluated set) and weights.npz.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

import npdl as nd
import models
from config import CACHE, DATA, FIGURES, HERE, RESULTS, RUNS

C = CACHE
OUT = RUNS
FS_OBS = 62_500.0
SYS_NAMES = json.load(open(f"{C}/syn_names.json"))["systemNames"]
REAL_MODS = sorted(SYS_NAMES)                     # 20-class label order shared by both datasets


def normalize(x):
    x = x - x.mean(axis=1, keepdims=True)
    rms = np.sqrt((x ** 2).sum(axis=2).mean(axis=1))
    return (x / np.maximum(rms, 1e-6)[:, None, None]).astype(np.float32)


def rotate(x, f_hz):
    """Multiply each complex window by exp(j 2 pi f n / fs); f_hz scalar or per-window array."""
    n = np.arange(x.shape[1], dtype=np.float64)
    ph = 2 * np.pi * np.outer(np.atleast_1d(f_hz), n) / FS_OBS
    c, s = np.cos(ph).astype(np.float32), np.sin(ph).astype(np.float32)
    i, q = x[..., 0], x[..., 1]
    return np.stack([i * c - q * s, i * s + q * c], axis=-1)


def syn_data():
    X = np.load(f"{C}/syn_X.npy", mmap_mode="r")
    m = pd.read_csv(f"{C}/syn_meta.csv")
    m["mod"] = [SYS_NAMES[int(s) - 1] if s >= 1 else "Idle" for s in m.y_system]
    return X, m


def real_data():
    return np.load(f"{C}/real_X.npy", mmap_mode="r"), pd.read_csv(f"{C}/real_meta.csv")


def labels(mods, states, ncls):
    """2 classes: 0 friendly, 1 adversarial. 20 classes: index into REAL_MODS (sorted names)."""
    if ncls == 2:
        return np.asarray(states, dtype=np.int64)
    return np.array([REAL_MODS.index(mm.replace("-", "_")) for mm in mods], dtype=np.int64)


def main(argv):
    kind = argv[0]
    t0 = time.time()
    X, m = syn_data()
    if kind == "syn":
        arch, seed = argv[1], int(argv[2])
        ncls = int(argv[3]) if len(argv) > 3 else 2
        job = f"syn_{arch}_s{seed}" + ("" if ncls == 2 else f"_{ncls}cls")
        occ = m.y_state > 0
        tr, va = (occ & (m.split == "train")).values, (occ & (m.split == "val")).values
        Xtr, Xva = normalize(X[tr]), normalize(X[va])
        ytr = labels(m["mod"][tr], m.y_state[tr] - 1, ncls)
        yva = labels(m["mod"][va], m.y_state[va] - 1, ncls)
        evalsets = {"test_all": (m.split == "test").values, "val_all": (m.split == "val").values}
        ev = {k: (normalize(X[v]), np.where(v)[0]) for k, v in evalsets.items()}
        n_real = 0
        cfo = 0.0
    else:
        held, regime, seed = argv[1], argv[2], int(argv[3])
        ncls = int(argv[4]) if len(argv) > 4 else 2
        cfo = float(argv[5]) if len(argv) > 5 else 0.0
        arch = "cnn_lstm"
        job = f"hw_{held}_{regime}_s{seed}" + ("" if ncls == 2 else f"_{ncls}cls") + (f"_cfo{cfo:g}k" if cfo else "")
        R, rm = real_data()
        rm["y"] = (rm.cls == "Adversarial").astype(int)
        trd = (rm.tx != held).values
        rtr, rva = trd & (rm.split == "train").values, trd & (rm.split == "validation").values
        Xtr, Xva = normalize(R[rtr]), normalize(R[rva])
        ytr = labels(rm.modulation[rtr], rm.y[rtr], ncls)
        yva = labels(rm.modulation[rva], rm.y[rva], ncls)
        n_real = len(Xtr)
        if regime == "combined":
            occ = m.y_state > 0
            str_, sva = (occ & (m.split == "train")).values, (occ & (m.split == "val")).values
            Xtr = np.concatenate([Xtr, normalize(X[str_])])
            Xva = np.concatenate([Xva, normalize(X[sva])])
            ytr = np.concatenate([ytr, labels(m["mod"][str_], m.y_state[str_] - 1, ncls)])
            yva = np.concatenate([yva, labels(m["mod"][sva], m.y_state[sva] - 1, ncls)])
        evalsets = {"heldout": (rm.tx == held).values, "indomain_test": trd & (rm.split == "test").values}
        ev = {k: (normalize(R[v]), np.where(v)[0]) for k, v in evalsets.items()}
    os.makedirs(f"{OUT}/{job}", exist_ok=True)
    log = open(f"{OUT}/{job}/train.log", "w")

    def p(s):
        print(s, file=log, flush=True)

    augment = None
    if cfo > 0:
        is_real = np.zeros(len(Xtr), bool)
        is_real[:n_real] = True

        def augment(xb, idx, rng):
            f = rng.uniform(-cfo * 1e3, cfo * 1e3, len(idx)) * is_real[idx]
            return rotate(xb, f)
    p(f"job {job}: train {Xtr.shape} val {Xva.shape} classes {np.bincount(ytr).tolist()}")
    model = models.build(arch, ncls, seed)
    info = models.train(model, Xtr, ytr, Xva, yva, seed, ncls=ncls, log=p, augment=augment)
    scores = {}
    for k, (Xe, rows) in ev.items():
        scores[f"{k}_prob"] = nd.softmax(models.predict_logits(model, Xe)).astype(np.float32)
        scores[f"{k}_rows"] = rows
    np.savez_compressed(f"{OUT}/{job}/scores.npz", **scores)
    np.savez_compressed(f"{OUT}/{job}/weights.npz", *model.get_state()[:len(model.params())],
                        *[a for b in model.bn_layers() for a in (b.rm, b.rv)])
    info.update({"job": job, "kind": kind, "arch": arch, "seed": seed, "ncls": ncls, "cfo_aug_khz": cfo,
                 "n_train": int(len(Xtr)), "n_train_real": int(n_real), "n_val": int(len(Xva)),
                 "n_params": model.n_params(), "wall_seconds": time.time() - t0,
                 "label_order": ["Friendly", "Adversarial"] if ncls == 2 else REAL_MODS})
    json.dump(info, open(f"{OUT}/{job}/info.json", "w"), indent=1)
    p(f"done {job} in {time.time() - t0:.0f}s, best val loss {info['best_val_loss']:.4f}, epochs {info['epochs']}")


if __name__ == "__main__":
    main(sys.argv[1:])
