"""Finite-difference gradient check of every architecture (float64, dropout disabled)."""
import numpy as np
import npdl as nd

nd.set_dtype(np.float64)
import models  # noqa: E402  (models picks up the dtype through npdl.DT at call time)


def check(name, ncls=2, B=4, n_entries=6, seed=0):
    model = models.build(name, ncls, seed)
    for l in model.layers:
        if isinstance(l, nd.Dropout):
            l.p = 0.0
        if isinstance(l, nd.Residual):
            for s in l.layers:
                if isinstance(s, nd.Dropout):
                    s.p = 0.0
    rng = np.random.default_rng(1)
    x = rng.standard_normal((B, 256, 2))
    y = rng.integers(0, ncls, B)
    w = np.linspace(0.8, 1.2, ncls)

    def loss_of():
        return nd.weighted_ce(model.forward(x, train=True), y, w)[0]

    for p in model.params():
        p.g[...] = 0
    loss, dl = nd.weighted_ce(model.forward(x, train=True), y, w)
    model.backward(dl)
    worst = 0.0
    for p in model.params():
        flat_idx = rng.choice(p.v.size, size=min(n_entries, p.v.size), replace=False)
        for fi in flat_idx:
            ix = np.unravel_index(fi, p.v.shape)
            old = p.v[ix]
            h = 1e-6
            p.v[ix] = old + h
            lp = loss_of()
            p.v[ix] = old - h
            lm = loss_of()
            p.v[ix] = old
            num = (lp - lm) / (2 * h)
            ana = p.g[ix]
            rel = abs(num - ana) / max(1e-7, abs(num) + abs(ana))
            if abs(num) + abs(ana) > 1e-9:
                worst = max(worst, rel)
    print(f"{name:12s} params {model.n_params():8d}  loss {loss:.4f}  worst relative error {worst:.2e}")
    return worst


if __name__ == "__main__":
    res = {n: check(n) for n in ["cnn", "cnn_lstm", "cnn_gru", "cldnn", "bilstm", "transformer"]}
    res["cnn_lstm_20cls"] = check("cnn_lstm", ncls=20)
    assert max(res.values()) < 1e-4, res
    print("all gradient checks passed")
