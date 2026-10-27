"""Architectures and MATLAB-style training loop for the replication study.

CNN-LSTM reproduces the layer list of train_real_cnn_lstm_31800_v1.m exactly (231,938
parameters for two classes). The other five architectures are specified here in full,
because the original MATLAB definitions are not part of the released code.
"""
import time
import numpy as np
import npdl as nd


def conv_stack(rng):
    """Three convolution blocks of the MATLAB CNN-LSTM: conv-BN-ReLU-pool, conv-BN-ReLU-pool, conv-BN-ReLU."""
    return [nd.Conv1D(2, 64, 7, rng), nd.BatchNorm(64), nd.ReLU(), nd.MaxPool1D(),
            nd.Conv1D(64, 128, 5, rng), nd.BatchNorm(128), nd.ReLU(), nd.MaxPool1D(),
            nd.Conv1D(128, 128, 3, rng), nd.BatchNorm(128), nd.ReLU()]


def head(nin, ncls, rng, drop_rng):
    return [nd.Dense(nin, 64, rng), nd.ReLU(), nd.Dropout(0.25, drop_rng), nd.Dense(64, ncls, rng)]


def build(name, ncls, seed):
    rng = np.random.default_rng(seed)                  # initialization
    drng = np.random.default_rng(seed + 10_000)        # dropout masks
    if name == "cnn_lstm":
        layers = conv_stack(rng) + [nd.Dropout(0.20, drng), nd.LSTM(128, 128, rng), nd.Dropout(0.35, drng)] \
            + head(128, ncls, rng, drng)
    elif name == "cnn_gru":
        layers = conv_stack(rng) + [nd.Dropout(0.20, drng), nd.GRU(128, 128, rng), nd.Dropout(0.35, drng)] \
            + head(128, ncls, rng, drng)
    elif name == "cnn":
        layers = conv_stack(rng) + [nd.Dropout(0.20, drng), nd.GlobalAvgPool()] + head(128, ncls, rng, drng)
    elif name == "cldnn":
        layers = conv_stack(rng) + [nd.Dropout(0.20, drng), nd.LSTM(128, 128, rng, return_seq=True),
                                    nd.LSTM(128, 128, rng), nd.Dropout(0.35, drng),
                                    nd.Dense(128, 128, rng), nd.ReLU(), nd.Dropout(0.25, drng)] \
            + head(128, ncls, rng, drng)
    elif name == "bilstm":
        layers = [nd.Patchify(4), nd.BiLSTM(8, 64, rng), nd.Dropout(0.35, drng)] + head(128, ncls, rng, drng)
    elif name == "transformer":
        d = 64
        block = lambda: [nd.Residual([nd.MHSA(d, 4, rng), nd.Dropout(0.10, drng)]), nd.LayerNorm(d),
                         nd.Residual([nd.Dense(d, 128, rng), nd.ReLU(), nd.Dense(128, d, rng),
                                      nd.Dropout(0.10, drng)]), nd.LayerNorm(d)]
        layers = [nd.Patchify(8), nd.Dense(16, d, rng), nd.PosEmbed(32, d, rng)] + block() + block() \
            + [nd.GlobalAvgPool(), nd.Dropout(0.25, drng)] + head(d, ncls, rng, drng)
    else:
        raise ValueError(name)
    return nd.Sequential(layers)


ARCH_NAMES = {"cnn": "CNN-only", "cnn_lstm": "CNN-LSTM", "cnn_gru": "CNN-GRU", "cldnn": "CLDNN",
              "bilstm": "BiLSTM-only", "transformer": "Residual Transformer"}


def class_weights(y, ncls):
    counts = np.bincount(y, minlength=ncls).astype(np.float64)
    w = len(y) / (ncls * np.maximum(counts, 1))
    return (w / w.mean()).astype(np.float32)


def predict_logits(model, X, bs=512):
    return np.concatenate([model.forward(X[i:i + bs], train=False) for i in range(0, len(X), bs)])


def finalize_bn(model, X, bs=512):
    """MATLAB 'population' statistics: average batch means and second moments over the training data."""
    bns = model.bn_layers()
    if not bns:
        return
    for b in bns:
        b.collect = [np.zeros_like(b.rm), np.zeros_like(b.rm), 0]
    for i in range(0, len(X), bs):
        model.forward(X[i:i + bs], train=False)
    for b in bns:
        s1, s2, n = b.collect
        b.rm[...] = s1 / n
        b.rv[...] = np.maximum(s2 / n - (s1 / n) ** 2, 0)
        b.collect = None


def train(model, Xtr, ytr, Xva, yva, seed, ncls=2, max_epochs=int(__import__('os').environ.get('MAX_EPOCHS', 40)), bs=128, lr0=1e-3, drop_period=8,
          drop_factor=0.5, l2=1e-4, clip=1.0, patience=8, augment=None, log=print):
    """trainingOptions('adam', ...) as in train_real_cnn_lstm_31800_v1.m.

    Validation runs every floor(iterations_per_epoch/2) iterations; training stops after `patience`
    validation checks without a new lowest validation loss; the lowest-loss network is returned.
    augment(xbatch, batch_indices, rng) -> xbatch is applied to training mini-batches only.
    """
    rng = np.random.default_rng(seed + 20_000)
    w = class_weights(ytr, ncls)
    params = model.params()
    n = len(Xtr)
    ipe = int(np.ceil(n / bs))
    vfreq = max(1, ipe // 2)
    b1, b2, eps = 0.9, 0.999, 1e-8
    it, best, best_state, bad, hist = 0, np.inf, None, 0, []
    t0 = time.time()
    stop = False
    for ep in range(max_epochs):
        lr = lr0 * drop_factor ** (ep // drop_period)
        order = rng.permutation(n)
        for bi in range(ipe):
            idx = order[bi * bs:(bi + 1) * bs]
            xb = Xtr[idx]
            if augment is not None:
                xb = augment(xb, idx, rng)
            for p in params:
                p.g[...] = 0
            logits = model.forward(xb, train=True)
            loss, dl = nd.weighted_ce(logits, ytr[idx], w)
            model.backward(dl)
            it += 1
            for p in params:
                if p.l2:
                    p.g += l2 * p.v
                gn = float(np.sqrt((p.g.astype(np.float64) ** 2).sum()))
                if gn > clip:
                    p.g *= clip / gn
                p.m = b1 * p.m + (1 - b1) * p.g
                p.s = b2 * p.s + (1 - b2) * p.g * p.g
                mh = p.m / (1 - b1 ** it)
                sh = p.s / (1 - b2 ** it)
                p.v -= (lr * mh / (np.sqrt(sh) + eps)).astype(p.v.dtype)
            if it % vfreq == 0:
                vl, _ = nd.weighted_ce(predict_logits(model, Xva), yva, w)
                hist.append((ep + 1, it, loss, vl))
                if vl < best:
                    best, best_state, bad = vl, model.get_state(), 0
                else:
                    bad += 1
                if bad >= patience:
                    stop = True
                    break
        log(f"  epoch {ep + 1:2d} lr {lr:.2e} train loss {loss:.4f} val loss {hist[-1][3]:.4f} "
            f"best {best:.4f} ({time.time() - t0:.0f}s)")
        if stop:
            break
    model.set_state(best_state)
    finalize_bn(model, Xtr)
    return {"best_val_loss": best, "epochs": hist[-1][0], "iterations": it, "seconds": time.time() - t0,
            "history": hist, "class_weights": w.tolist()}
