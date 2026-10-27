"""Stage 1 analysis on Dataset v5: theory check, noise-uncertainty sensitivity, CFAR alternative.

Energy statistic E = (1/N) sum |y|^2 with N = 256. Idle windows are unit-variance circular complex
Gaussian noise, so 2N E ~ chi2(2N) under H0 and, for a window with full-band SNR rho, approximately
noncentral chi2(2N, 2N rho) under H1.

Noise-power uncertainty is emulated by scaling each test window by a factor g = 10^(u/10), u uniform
in [-U, U] dB, which multiplies signal and noise power alike (an unknown receiver gain or noise level).
The fixed detector keeps the absolute threshold 1.108. The noise-referenced (CFAR) detector divides E
by a noise-power estimate from K reference samples drawn with the same g (as the hardware noise-only
buffer would provide) and uses a threshold from the F(2N, 2K) distribution for a target P_FA.
"""
import json
import numpy as np, pandas as pd, scipy.io as sio
from scipy import stats
from config import CACHE, DATA, FIGURES, HERE, RESULTS, RUNS

C, OUT = CACHE, RESULTS
N, K = 256, 4096
m = pd.read_csv(f"{C}/syn_meta.csv")
E = np.load(f"{C}/syn_energy.npy").astype(np.float64)
thr = float(sio.loadmat(f"{DATA}/energy_gate_v5.mat", squeeze_me=True)["threshold"])
te = (m.split == "test").values
y = m.y_state.values
idle, occ = (y == 0), (y > 0)
snr = m.snr_db.values
res = {"threshold": thr, "N": N, "K_reference": K}

# Neyman-Pearson thresholds for known unit noise
res["pfa_theory_at_threshold"] = float(stats.chi2.sf(2 * N * thr, 2 * N))
res["np_threshold_for_pfa_5pct"] = float(stats.chi2.isf(0.05, 2 * N) / (2 * N))

# CFAR threshold for the same nominal P_FA as the fixed detector's theory value
pfa_target = res["pfa_theory_at_threshold"]
gamma = float(stats.f.isf(pfa_target, 2 * N, 2 * K))
res["cfar_gamma"] = gamma
res["cfar_target_pfa"] = pfa_target

# sensitivity to the SNR prior: balanced-accuracy-optimal threshold under different priors (training split)
tr = (m.split == "train").values
grid = np.linspace(0.95, 1.4, 1801)


def ba_opt(mask_occ):
    pf = np.array([np.mean(E[tr & idle] > t) for t in grid])
    pdv = np.array([np.mean(E[tr & mask_occ] > t) for t in grid])
    i = np.argmax(0.5 * (1 - pf) + 0.5 * pdv)
    return float(grid[i]), float(pf[i])


res["ba_threshold_prior_uniform_-20_25"] = ba_opt(occ)
res["ba_threshold_prior_uniform_-10_25"] = ba_opt(occ & (snr >= -10))
res["ba_threshold_prior_uniform_0_25"] = ba_opt(occ & (snr >= 0))

# noise-uncertainty Monte Carlo on the test windows
rng = np.random.default_rng(7)
rows = []
for U in [0.0, 0.25, 0.5, 1.0, 2.0]:
    pf_fixed, pf_cfar, pd_fixed, pd_cfar = [], [], [], []
    for rep in range(20):
        u = rng.uniform(-U, U, te.sum())
        g = 10 ** (u / 10)
        Eg = E[te] * g
        sig2 = g * rng.chisquare(2 * K, te.sum()) / (2 * K)
        yi = y[te]
        fixed = Eg > thr
        cfar = Eg / sig2 > gamma
        pf_fixed.append(np.mean(fixed[yi == 0]))
        pf_cfar.append(np.mean(cfar[yi == 0]))
        pd_fixed.append(np.mean(fixed[yi > 0]))
        pd_cfar.append(np.mean(cfar[yi > 0]))
    rows.append({"U_dB": U, "pfa_fixed": np.mean(pf_fixed), "pfa_cfar": np.mean(pf_cfar),
                 "pd_fixed": np.mean(pd_fixed), "pd_cfar": np.mean(pd_cfar)})
unc = pd.DataFrame(rows)
print(unc.round(4).to_string(index=False))

# fixed-offset curve: P_FA and P_D when the noise power is off by a constant u dB
off = []
for u in np.linspace(-2, 2, 81):
    g = 10 ** (u / 10)
    off.append({"u_dB": u,
                "pfa_fixed_theory": stats.chi2.sf(2 * N * thr / g, 2 * N),
                "pfa_cfar_theory": pfa_target,
                "pfa_fixed_measured": np.mean(E[te & idle] * g > thr)})
off = pd.DataFrame(off)

# detection probability versus SNR: measured (fixed threshold), known-noise theory, CFAR theory
pdrows = []
for s0 in np.arange(-20, 25.01, 1.0):
    r = 10 ** (s0 / 10)
    k = te & occ & (np.abs(snr - s0) <= 1.25)
    pdrows.append({"snr_db": s0, "pd_measured": float(np.mean(E[k] > thr)) if k.any() else np.nan,
                   "n": int(k.sum()),
                   "pd_theory_known_noise": float(stats.ncx2.sf(2 * N * thr, 2 * N, 2 * N * r)),
                   "pd_theory_cfar": float(stats.ncf.sf(gamma, 2 * N, 2 * K, 2 * N * r))})
pdc = pd.DataFrame(pdrows)
print(pdc[pdc.snr_db.isin([-20, -15, -10, -8, -6, -4])].round(4).to_string(index=False))
res["detectability_N_for_pd90_pfa05_at_-20dB"] = float(((stats.norm.isf(0.05) - stats.norm.isf(0.9) * 1.01) / 0.01) ** 2)
res["detectability_N_for_pd90_pfa05_at_-10dB"] = float(((stats.norm.isf(0.05) - stats.norm.isf(0.9) * 1.1) / 0.1) ** 2)
for k_, v in res.items():
    print(f"{k_}: {v}")
import os
os.makedirs(OUT, exist_ok=True)
unc.to_csv(f"{OUT}/stage1_noise_uncertainty.csv", index=False)
off.to_csv(f"{OUT}/stage1_noise_offset.csv", index=False)
pdc.to_csv(f"{OUT}/stage1_pd_vs_snr.csv", index=False)
json.dump(res, open(f"{OUT}/stage1_summary.json", "w"), indent=1)
