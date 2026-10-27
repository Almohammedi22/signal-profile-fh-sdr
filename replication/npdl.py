"""Minimal NumPy deep-learning library used to re-implement the MATLAB classifiers.

Layout is channels-last: sequences are (batch, time, features). Every layer implements
forward(x, train) and backward(dy); parameters carry their gradient and Adam state.
Defaults follow MATLAB Deep Learning Toolbox conventions where they matter:
  * Glorot-uniform weights for convolution, fully connected and recurrent input weights,
    orthogonal recurrent weights, zero biases, forget-gate bias 1 ("unit-forget-gate");
  * batch normalization with epsilon 1e-5 and population statistics finalized after training;
  * L2 regularization added to the gradient of weights, batch-norm scale and offset
    (bias L2 factor 0), then per-parameter L2-norm gradient clipping, then Adam.
"""
import numpy as np

DT = np.float32


def set_dtype(dt):
    global DT
    DT = dt


class Param:
    def __init__(self, value, l2=True):
        self.v = np.asarray(value, dtype=DT)
        self.g = np.zeros_like(self.v)
        self.l2 = l2
        self.m = np.zeros_like(self.v)
        self.s = np.zeros_like(self.v)


def glorot(rng, shape, fan_in, fan_out):
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-a, a, shape)


def orthogonal(rng, rows, cols):
    """rows x cols matrix with orthonormal columns (rows >= cols), as MATLAB 'orthogonal'."""
    z = rng.standard_normal((rows, cols))
    q, r = np.linalg.qr(z)
    return q * np.sign(np.diag(r))


def sigmoid(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


class Layer:
    def params(self):
        return []

    def forward(self, x, train):
        raise NotImplementedError

    def backward(self, dy):
        raise NotImplementedError


class Conv1D(Layer):
    """'same'-padded 1-D convolution, stride 1, odd kernel."""

    def __init__(self, cin, cout, k, rng):
        self.k, self.cin, self.cout, self.p = k, cin, cout, (k - 1) // 2
        self.W = Param(glorot(rng, (k * cin, cout), k * cin, k * cout))
        self.b = Param(np.zeros(cout), l2=False)

    def params(self):
        return [self.W, self.b]

    def forward(self, x, train):
        B, L, C = x.shape
        xp = np.pad(x, ((0, 0), (self.p, self.p), (0, 0)))
        win = np.lib.stride_tricks.sliding_window_view(xp, self.k, axis=1)  # B,L,C,k
        cols = np.ascontiguousarray(win.transpose(0, 1, 3, 2)).reshape(B * L, self.k * C)
        self.cache = (cols, B, L, C)
        return (cols @ self.W.v + self.b.v).reshape(B, L, self.cout)

    def backward(self, dy):
        cols, B, L, C = self.cache
        d2 = dy.reshape(B * L, self.cout)
        self.W.g += cols.T @ d2
        self.b.g += d2.sum(0)
        dcols = (d2 @ self.W.v.T).reshape(B, L, self.k, C)
        dxp = np.zeros((B, L + 2 * self.p, C), DT)
        for j in range(self.k):
            dxp[:, j:j + L, :] += dcols[:, :, j, :]
        return dxp[:, self.p:self.p + L, :]


class BatchNorm(Layer):
    """Normalizes over every axis except the last (channels)."""

    def __init__(self, c, eps=1e-5, momentum=0.1):
        self.gamma = Param(np.ones(c))
        self.beta = Param(np.zeros(c))
        self.rm = np.zeros(c, DT)
        self.rv = np.ones(c, DT)
        self.eps, self.mom = eps, momentum
        self.collect = None

    def params(self):
        return [self.gamma, self.beta]

    def forward(self, x, train):
        axes = tuple(range(x.ndim - 1))
        if train or self.collect is not None:
            mu = x.mean(axes)
            var = x.var(axes)
            if self.collect is not None and not train:      # population-statistics pass
                self.collect[0] += mu
                self.collect[1] += (x * x).mean(axes)
                self.collect[2] += 1
            inv = 1.0 / np.sqrt(var + self.eps)
            xhat = (x - mu) * inv
            if train:
                n = x.size // x.shape[-1]
                self.rm = (1 - self.mom) * self.rm + self.mom * mu
                self.rv = (1 - self.mom) * self.rv + self.mom * var * (n / max(n - 1, 1))
                self.cache = (xhat, inv, axes, n)
            return xhat * self.gamma.v + self.beta.v
        return (x - self.rm) / np.sqrt(self.rv + self.eps) * self.gamma.v + self.beta.v

    def backward(self, dy):
        xhat, inv, axes, n = self.cache
        self.gamma.g += (dy * xhat).sum(axes)
        self.beta.g += dy.sum(axes)
        dxh = dy * self.gamma.v
        return (inv / n) * (n * dxh - dxh.sum(axes) - xhat * (dxh * xhat).sum(axes))


class ReLU(Layer):
    def forward(self, x, train):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dy):
        return dy * self.mask


class MaxPool1D(Layer):
    """Pool size 2, stride 2 along time."""

    def forward(self, x, train):
        B, L, C = x.shape
        r = x[:, :L // 2 * 2].reshape(B, L // 2, 2, C)
        self.arg = r.argmax(2)
        self.shape = x.shape
        return r.max(2)

    def backward(self, dy):
        B, L, C = self.shape
        dx = np.zeros((B, L // 2, 2, C), DT)
        np.put_along_axis(dx, self.arg[:, :, None, :], dy[:, :, None, :], axis=2)
        out = np.zeros(self.shape, DT)
        out[:, :L // 2 * 2] = dx.reshape(B, L // 2 * 2, C)
        return out


class Dropout(Layer):
    def __init__(self, p, rng):
        self.p, self.rng = p, rng

    def forward(self, x, train):
        if not train or self.p == 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape, dtype=np.float32) >= self.p).astype(DT) / (1 - self.p)
        return x * self.mask

    def backward(self, dy):
        return dy if self.mask is None else dy * self.mask


class Dense(Layer):
    """Applies to the last axis; works on (B, F) or (B, T, F)."""

    def __init__(self, nin, nout, rng):
        self.W = Param(glorot(rng, (nin, nout), nin, nout))
        self.b = Param(np.zeros(nout), l2=False)

    def params(self):
        return [self.W, self.b]

    def forward(self, x, train):
        self.x = x
        return x @ self.W.v + self.b.v

    def backward(self, dy):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        d2 = dy.reshape(-1, dy.shape[-1])
        self.W.g += x2.T @ d2
        self.b.g += d2.sum(0)
        return dy @ self.W.v.T


class GlobalAvgPool(Layer):
    def forward(self, x, train):
        self.T = x.shape[1]
        return x.mean(1)

    def backward(self, dy):
        return np.repeat(dy[:, None, :] / self.T, self.T, axis=1)


class LSTM(Layer):
    """Gate order (input, forget, cell candidate, output) as in MATLAB."""

    def __init__(self, nin, h, rng, return_seq=False, reverse=False):
        self.h, self.return_seq, self.reverse = h, return_seq, reverse
        self.Wx = Param(glorot(rng, (nin, 4 * h), nin, 4 * h))
        self.Wh = Param(orthogonal(rng, 4 * h, h).T)
        b = np.zeros(4 * h)
        b[h:2 * h] = 1.0
        self.b = Param(b, l2=False)

    def params(self):
        return [self.Wx, self.Wh, self.b]

    def forward(self, x, train):
        if self.reverse:
            x = x[:, ::-1]
        B, T, F = x.shape
        H = self.h
        self.x = x
        Xp = (x.reshape(B * T, F) @ self.Wx.v + self.b.v).reshape(B, T, 4 * H)
        hs = np.zeros((T + 1, B, H), DT)
        cs = np.zeros((T + 1, B, H), DT)
        G = np.empty((T, B, 4 * H), DT)
        TC = np.empty((T, B, H), DT)
        for t in range(T):
            z = Xp[:, t] + hs[t] @ self.Wh.v
            g = G[t]
            np.multiply(z, 0.5, out=g)
            g[:, 2 * H:3 * H] = z[:, 2 * H:3 * H]
            np.tanh(g, out=g)
            g[:, :2 * H] += 1.0
            g[:, :2 * H] *= 0.5
            g[:, 3 * H:] += 1.0
            g[:, 3 * H:] *= 0.5
            i, f, gg, o = g[:, :H], g[:, H:2 * H], g[:, 2 * H:3 * H], g[:, 3 * H:]
            c = cs[t + 1]
            np.multiply(f, cs[t], out=c)
            c += i * gg
            np.tanh(c, out=TC[t])
            np.multiply(o, TC[t], out=hs[t + 1])
        self.cache = (hs, cs, G, TC)
        if self.return_seq:
            out = hs[1:].transpose(1, 0, 2)
            return out[:, ::-1] if self.reverse else out
        return hs[T].copy()

    def backward(self, dy):
        hs, cs, G, TC = self.cache
        x = self.x
        B, T, F = x.shape
        H = self.h
        if self.return_seq:
            dseq = dy[:, ::-1] if self.reverse else dy
            dh = np.zeros((B, H), DT)
        else:
            dseq = None
            dh = dy.astype(DT, copy=True)
        dc = np.zeros((B, H), DT)
        dZ = np.empty((T, B, 4 * H), DT)
        WhT = self.Wh.v.T
        for t in range(T - 1, -1, -1):
            if dseq is not None:
                dh = dh + dseq[:, t]
            g = G[t]
            i, f, gg, o = g[:, :H], g[:, H:2 * H], g[:, 2 * H:3 * H], g[:, 3 * H:]
            tc = TC[t]
            dc += dh * o * (1.0 - tc * tc)
            dz = dZ[t]
            np.multiply(dc * gg, i * (1.0 - i), out=dz[:, :H])
            np.multiply(dc * cs[t], f * (1.0 - f), out=dz[:, H:2 * H])
            np.multiply(dc * i, 1.0 - gg * gg, out=dz[:, 2 * H:3 * H])
            np.multiply(dh * tc, o * (1.0 - o), out=dz[:, 3 * H:])
            dh = dz @ WhT
            dc *= f
        self.Wh.g += hs[:-1].reshape(T * B, H).T @ dZ.reshape(T * B, 4 * H)
        dXp = dZ.transpose(1, 0, 2).reshape(B * T, 4 * H)
        self.Wx.g += x.reshape(B * T, F).T @ dXp
        self.b.g += dXp.sum(0)
        dx = (dXp @ self.Wx.v.T).reshape(B, T, F)
        return dx[:, ::-1] if self.reverse else dx


class GRU(Layer):
    """Reset gate applied after the recurrent multiplication (MATLAB default, cuDNN style)."""

    def __init__(self, nin, h, rng, return_seq=False):
        self.h, self.return_seq = h, return_seq
        self.Wx = Param(glorot(rng, (nin, 3 * h), nin, 3 * h))
        self.Wh = Param(orthogonal(rng, 3 * h, h).T)
        self.bx = Param(np.zeros(3 * h), l2=False)
        self.bh = Param(np.zeros(3 * h), l2=False)

    def params(self):
        return [self.Wx, self.Wh, self.bx, self.bh]

    def forward(self, x, train):
        B, T, F = x.shape
        H = self.h
        self.x = x
        Xp = (x.reshape(B * T, F) @ self.Wx.v + self.bx.v).reshape(B, T, 3 * H)
        hs = np.zeros((T + 1, B, H), DT)
        R = np.empty((T, B, H), DT)
        Z = np.empty((T, B, H), DT)
        N = np.empty((T, B, H), DT)
        HPN = np.empty((T, B, H), DT)
        for t in range(T):
            hp = hs[t] @ self.Wh.v + self.bh.v
            r = sigmoid(Xp[:, t, :H] + hp[:, :H])
            z = sigmoid(Xp[:, t, H:2 * H] + hp[:, H:2 * H])
            n = np.tanh(Xp[:, t, 2 * H:] + r * hp[:, 2 * H:])
            R[t], Z[t], N[t], HPN[t] = r, z, n, hp[:, 2 * H:]
            hs[t + 1] = n + z * (hs[t] - n)
        self.cache = (hs, R, Z, N, HPN)
        if self.return_seq:
            return hs[1:].transpose(1, 0, 2)
        return hs[T].copy()

    def backward(self, dy):
        hs, R, Z, N, HPN = self.cache
        x = self.x
        B, T, F = x.shape
        H = self.h
        dseq = dy if self.return_seq else None
        dh = np.zeros((B, H), DT) if self.return_seq else dy.astype(DT, copy=True)
        dXp = np.empty((T, B, 3 * H), DT)
        dHp = np.empty((T, B, 3 * H), DT)
        WhT = self.Wh.v.T
        for t in range(T - 1, -1, -1):
            if dseq is not None:
                dh = dh + dseq[:, t]
            r, z, n = R[t], Z[t], N[t]
            dn = dh * (1.0 - z)
            dzv = dh * (hs[t] - n)
            dnp = dn * (1.0 - n * n)
            dr = dnp * HPN[t]
            drp = dr * r * (1.0 - r)
            dzp = dzv * z * (1.0 - z)
            dXp[t, :, :H] = drp
            dXp[t, :, H:2 * H] = dzp
            dXp[t, :, 2 * H:] = dnp
            dHp[t, :, :H] = drp
            dHp[t, :, H:2 * H] = dzp
            dHp[t, :, 2 * H:] = dnp * r
            dh = dh * z + dHp[t] @ WhT
        self.Wh.g += hs[:-1].reshape(T * B, H).T @ dHp.reshape(T * B, 3 * H)
        self.bh.g += dHp.sum((0, 1))
        dX2 = dXp.transpose(1, 0, 2).reshape(B * T, 3 * H)
        self.Wx.g += x.reshape(B * T, F).T @ dX2
        self.bx.g += dX2.sum(0)
        return (dX2 @ self.Wx.v.T).reshape(B, T, F)


class BiLSTM(Layer):
    """Concatenates the final state of the forward pass (after t=T) and of the backward pass (after t=1)."""

    def __init__(self, nin, h, rng):
        self.f = LSTM(nin, h, rng)
        self.r = LSTM(nin, h, rng, reverse=True)
        self.h = h

    def params(self):
        return self.f.params() + self.r.params()

    def forward(self, x, train):
        return np.concatenate([self.f.forward(x, train), self.r.forward(x, train)], axis=1)

    def backward(self, dy):
        return self.f.backward(dy[:, :self.h]) + self.r.backward(dy[:, self.h:])


class Patchify(Layer):
    """(B, L, C) -> (B, L/P, P*C) tokens."""

    def __init__(self, p):
        self.p = p

    def forward(self, x, train):
        B, L, C = x.shape
        self.shape = x.shape
        return x.reshape(B, L // self.p, self.p * C)

    def backward(self, dy):
        return dy.reshape(self.shape)


class PosEmbed(Layer):
    def __init__(self, t, d, rng):
        self.P = Param(rng.normal(0, 0.02, (t, d)))

    def params(self):
        return [self.P]

    def forward(self, x, train):
        return x + self.P.v

    def backward(self, dy):
        self.P.g += dy.sum(0)
        return dy


class LayerNorm(Layer):
    def __init__(self, d, eps=1e-5):
        self.gamma = Param(np.ones(d))
        self.beta = Param(np.zeros(d))
        self.eps = eps

    def params(self):
        return [self.gamma, self.beta]

    def forward(self, x, train):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + self.eps)
        xh = (x - mu) * inv
        self.cache = (xh, inv)
        return xh * self.gamma.v + self.beta.v

    def backward(self, dy):
        xh, inv = self.cache
        d = xh.shape[-1]
        self.gamma.g += (dy * xh).reshape(-1, d).sum(0)
        self.beta.g += dy.reshape(-1, d).sum(0)
        dxh = dy * self.gamma.v
        return (inv / d) * (d * dxh - dxh.sum(-1, keepdims=True) - xh * (dxh * xh).sum(-1, keepdims=True))


class MHSA(Layer):
    def __init__(self, d, heads, rng):
        self.d, self.hn, self.dk = d, heads, d // heads
        self.Wq, self.Wk, self.Wv, self.Wo = (Dense(d, d, rng) for _ in range(4))

    def params(self):
        return self.Wq.params() + self.Wk.params() + self.Wv.params() + self.Wo.params()

    def _split(self, x):
        B, T, _ = x.shape
        return x.reshape(B, T, self.hn, self.dk).transpose(0, 2, 1, 3)

    def forward(self, x, train):
        B, T, _ = x.shape
        q, k, v = (self._split(m.forward(x, train)) for m in (self.Wq, self.Wk, self.Wv))
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        s -= s.max(-1, keepdims=True)
        a = np.exp(s)
        a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, self.d)
        self.cache = (q, k, v, a)
        return self.Wo.forward(o, train)

    def backward(self, dy):
        q, k, v, a = self.cache
        B, _, T, _ = q.shape
        do = self._split(self.Wo.backward(dy))
        da = do @ v.transpose(0, 1, 3, 2)
        dv = a.transpose(0, 1, 3, 2) @ do
        ds = a * (da - (da * a).sum(-1, keepdims=True)) / np.sqrt(self.dk)
        dq = ds @ k
        dk = ds.transpose(0, 1, 3, 2) @ q
        merge = lambda t: t.transpose(0, 2, 1, 3).reshape(B, T, self.d)
        return self.Wq.backward(merge(dq)) + self.Wk.backward(merge(dk)) + self.Wv.backward(merge(dv))


class Residual(Layer):
    """y = x + f(x) for a list of layers f."""

    def __init__(self, layers):
        self.layers = layers

    def params(self):
        return [p for l in self.layers for p in l.params()]

    def forward(self, x, train):
        y = x
        for l in self.layers:
            y = l.forward(y, train)
        return x + y

    def backward(self, dy):
        g = dy
        for l in reversed(self.layers):
            g = l.backward(g)
        return dy + g


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def params(self):
        return [p for l in self.layers for p in l.params()]

    def forward(self, x, train=False):
        for l in self.layers:
            x = l.forward(x, train)
        return x

    def backward(self, dy):
        for l in reversed(self.layers):
            dy = l.backward(dy)
        return dy

    def bn_layers(self):
        out = []

        def walk(ls):
            for l in ls:
                if isinstance(l, BatchNorm):
                    out.append(l)
                if isinstance(l, Residual):
                    walk(l.layers)
        walk(self.layers)
        return out

    def n_params(self):
        return int(sum(p.v.size for p in self.params()))

    def get_state(self):
        return [p.v.copy() for p in self.params()] + [(b.rm.copy(), b.rv.copy()) for b in self.bn_layers()]

    def set_state(self, st):
        ps = self.params()
        for p, v in zip(ps, st[:len(ps)]):
            p.v[...] = v
        for b, (m, v) in zip(self.bn_layers(), st[len(ps):]):
            b.rm[...] = m
            b.rv[...] = v


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def weighted_ce(logits, y, w):
    """MATLAB weighted cross-entropy: -(1/N) sum_n w[y_n] log p[n, y_n]. Returns loss and dlogits."""
    p = softmax(logits)
    n = len(y)
    wy = w[y]
    loss = float(-(wy * np.log(np.maximum(p[np.arange(n), y], 1e-12))).sum() / n)
    d = p.copy()
    d[np.arange(n), y] -= 1.0
    d *= (wy / n)[:, None]
    return loss, d.astype(DT)
