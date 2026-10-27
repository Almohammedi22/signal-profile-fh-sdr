"""Properties of synthetic Dataset v5 measured on the windows themselves (no generator code needed).

For high-SNR occupied windows (SNR >= 15 dB):
  * carrier offset: spectral line of y^2 (BPSK, PAM, ASK) or y^4 (QPSK) -> frequency offset in cycles/sample;
  * symbol period: strongest cyclic frequency of |y|^2 in (0, 0.5) cycles/sample, averaged over windows;
  * improperness |E[y^2]|/E[|y|^2] for QPSK/16QAM windows (IQ imbalance indicator; 0 for proper signals);
  * dependence of the spectral centroid on the hop channel index (Spearman correlation);
  * per-receiver (rx_id) and per-hop-pattern summary of these quantities.
"""
import json, sys
import numpy as np, pandas as pd
from scipy import stats
from config import CACHE, DATA, FIGURES, HERE, RESULTS, RUNS

C = CACHE
X = np.load(f"{C}/syn_X.npy", mmap_mode="r")
m = pd.read_csv(f"{C}/syn_meta.csv")
names = json.load(open(f"{C}/syn_names.json"))["systemNames"]
m["mod"] = [names[int(s) - 1] if s >= 1 else "Idle" for s in m.y_system]
hi = (m.y_state > 0) & (m.snr_db >= 15)
NF = 4096
fr = np.fft.fftfreq(NF)
out = {}

def z_of(i):
    a = np.asarray(X[i], dtype=np.float64)
    return a[:, 0] + 1j * a[:, 1]

# carrier offset from power-law spectral lines
for mod, pw in [("BPSK", 2), ("PAM4", 2), ("PAM8", 2), ("ASK", 2), ("QPSK", 4), ("8PSK", 8)]:
    idx = np.where(hi & (m["mod"] == mod))[0]
    f = []
    for i in idx:
        z = z_of(i)
        if mod == "ASK":
            z = z - z.mean()
        S = np.abs(np.fft.fft(z ** pw, NF))
        f.append(fr[S.argmax()] / pw)
    f = np.array(f)
    out[f"offset_{mod}"] = {"n": int(len(f)), "median_cyc": float(np.median(f)), "iqr_cyc": float(np.subtract(*np.percentile(f, [75, 25]))),
                           "frac_abs_lt_1e-3": float(np.mean(np.abs(f) < 1e-3))}

# cyclic frequency of |y|^2 (symbol rate) per modulation
cyc = {}
for mod in ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "GFSK", "CPFSK", "MSK", "PAM4", "PAM8", "ASK", "OOK", "DSSS", "FHSS"]:
    idx = np.where(hi & (m["mod"] == mod))[0][:150]
    acc = np.zeros(NF)
    for i in idx:
        z = z_of(i)
        p = np.abs(z) ** 2
        acc += np.abs(np.fft.fft(p - p.mean(), NF)) ** 2
    pos = (fr > 0.02) & (fr < 0.5)
    k = np.argmax(acc * pos)
    peak = acc[k] / np.median(acc[pos])
    cyc[mod] = {"cyclic_freq": float(fr[k]), "period_samples": float(1 / fr[k]), "peak_to_median": float(peak)}
out["cyclic_|y|^2"] = cyc

# improperness for QPSK and 16QAM
for mod in ["QPSK", "16QAM", "64QAM", "OFDM"]:
    idx = np.where(hi & (m["mod"] == mod))[0]
    v = []
    for i in idx:
        z = z_of(i)
        z = z - z.mean()
        v.append(np.abs(np.mean(z ** 2)) / np.mean(np.abs(z) ** 2))
    out[f"improper_{mod}"] = {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}

# spectral centroid vs channel index, and per rx_id
occ = np.where((m.y_state > 0) & (m.snr_db >= 10))[0]
cent = np.empty(len(occ))
bw = np.empty(len(occ))
for j, i in enumerate(occ):
    z = z_of(i)
    S = np.abs(np.fft.fft(z, 256)) ** 2
    f256 = np.fft.fftfreq(256)
    cent[j] = np.sum(f256 * S) / S.sum()
    o = np.argsort(np.fft.fftshift(S))[::-1]
    bw[j] = np.searchsorted(np.cumsum(np.fft.fftshift(S)[o]) / S.sum(), 0.9) / 256
sub = m.iloc[occ].assign(centroid=cent, bw90=bw)
rho, pval = stats.spearmanr(sub.channel_id, sub.centroid)
out["centroid_vs_channel_spearman"] = {"rho": float(rho), "p": float(pval), "n": int(len(sub))}
out["per_rx"] = sub.groupby("rx_id").agg(n=("centroid", "size"), centroid_med=("centroid", "median"), bw90_med=("bw90", "median")).round(4).to_dict(orient="index")
out["rx_counts_by_state"] = pd.crosstab(m.rx_id, m.y_state).to_dict(orient="index")
out["bw90_by_mod"] = sub.groupby("mod").bw90.median().round(3).to_dict()
# per-mod bandwidth vs channel (does channel change the waveform?)
g = []
for mod, d in sub.groupby("mod"):
    if len(d) > 30:
        r, p = stats.spearmanr(d.channel_id, d.bw90)
        g.append((mod, round(float(r), 3), round(float(p), 3)))
out["bw90_vs_channel_spearman_by_mod"] = g
json.dump(out, open(f"{RESULTS}/synthetic_measured_properties.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:5000])
