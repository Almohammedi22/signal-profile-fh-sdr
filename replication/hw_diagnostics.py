"""Hardware diagnostics for the alternative explanations of the Pluto_05 PAM4/PAM8 errors.

For every capture with a real-valued constellation (BPSK, ASK, OOK, PAM4, PAM8):
  * total frequency from the squared-signal spectral line, then de-rotation and phase alignment;
  * received power relative to the same transmitter's BPSK captures (transmit-chain response at the
    35-55 kHz digital subcarriers, after the receiver's 16:1 averaging);
  * PAM level structure: 1-D k-means of the aligned real amplitudes, level spacing of the outer pair
    relative to the inner pairs (gain compression would shrink the outer spacing), and per-symbol SNR;
  * ADC headroom from the stored maximum raw amplitude (full scale 2048).
"""
import json, sys
import numpy as np, pandas as pd

from run_experiment import real_data
from config import CACHE, DATA, FIGURES, HERE, RESULTS, RUNS

FS = 62_500.0
SUB = {"BPSK": 0.0, "ASK": 55e3, "OOK": 45e3, "PAM4": 40e3, "PAM8": 35e3}
CFO = {"Pluto_02": -1387.0, "Pluto_04": -658.0, "Pluto_05": -2404.0}
alias = lambda f: (f + FS / 2) % FS - FS / 2


def kmeans1d(v, k, iters=50):
    c = np.quantile(v, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        lab = np.abs(v[:, None] - c[None]).argmin(1)
        c = np.array([v[lab == j].mean() if (lab == j).any() else c[j] for j in range(k)])
    lab = np.abs(v[:, None] - c[None]).argmin(1)
    resid = v - c[lab]
    return np.sort(c), resid


def main():
    R, rm = real_data()
    gt = pd.read_csv(f"{DATA}/ground_truth_31800.csv",
                     low_memory=False)
    gt = gt.set_index("capture_order").loc[rm.order.values].reset_index()
    assert (gt.tx_unit.values == rm.tx.values).all() and (gt.modulation_name.values == rm.modulation.values).all()
    n = np.arange(256)
    rows = []
    for i in np.where(rm.modulation.isin(list(SUB)).values)[0]:
        mod, tx = rm.modulation[i], rm.tx[i]
        z = R[i, :, 0].astype(np.float64) + 1j * R[i, :, 1]
        z = z - z.mean()
        f_exp = alias(SUB[mod] + CFO[tx])
        S = np.abs(np.fft.fft(z ** 2, 4096))
        fr = np.fft.fftfreq(4096, 1 / FS)
        line = fr[S.argmax()]
        cands = [(line + k * FS) / 2 for k in range(-3, 4)]
        f = min(cands, key=lambda c: abs(alias(c - f_exp)))
        zd = z * np.exp(-2j * np.pi * f * n / FS)
        phi = 0.5 * np.angle((zd ** 2).sum())
        y = (zd * np.exp(-1j * phi)).real
        q = (zd * np.exp(-1j * phi)).imag
        rec = {"row": i, "tx": tx, "mod": mod, "power": float((np.abs(z) ** 2).mean()),
               "max_amp": float(gt.maximum_amplitude[i]), "freq_hz": float(f)}
        if mod in ("PAM4", "PAM8", "BPSK"):
            k = {"PAM4": 4, "PAM8": 8, "BPSK": 2}[mod]
            lv, resid = kmeans1d(y, k)
            sp = np.diff(lv)
            rec["outer_to_inner_spacing"] = float(0.5 * (sp[0] + sp[-1]) / sp[1:-1].mean()) if k > 2 else np.nan
            rec["level_snr_db"] = float(10 * np.log10(np.mean(y ** 2) / np.mean(resid ** 2)))
            rec["quadrature_fraction"] = float(np.mean(q ** 2) / np.mean(y ** 2 + q ** 2))
        rows.append(rec)
    d = pd.DataFrame(rows)
    bp = d[d["mod"] == "BPSK"].groupby("tx").power.median()
    d["power_rel_bpsk_db"] = 10 * np.log10(d.power / d.tx.map(bp))
    out = {}
    for (tx, mod), g in d.groupby(["tx", "mod"]):
        out[f"{tx}/{mod}"] = {"power_rel_bpsk_db_median": round(float(g.power_rel_bpsk_db.median()), 2),
                              "outer_to_inner_spacing_median": round(float(g.outer_to_inner_spacing.median()), 3)
                              if g.outer_to_inner_spacing.notna().any() else None,
                              "level_snr_db_median": round(float(g.level_snr_db.median()), 2)
                              if "level_snr_db" in g and g.level_snr_db.notna().any() else None,
                              "quadrature_fraction_median": round(float(g.quadrature_fraction.median()), 4)
                              if "quadrature_fraction" in g and g.quadrature_fraction.notna().any() else None,
                              "observed_freq_hz_median": round(float(g.freq_hz.median()), 1)}
    adc = rm.assign(max_amp=gt.maximum_amplitude.values).groupby("tx").max_amp.describe(percentiles=[0.5, 0.99])
    out["adc_max_amplitude"] = adc.round(1).to_dict(orient="index")
    snr = gt.assign(tx=rm.tx.values).groupby("tx")[["capture_to_noise_db", "window_to_noise_db"]].median()
    out["median_capture_to_noise_db"] = snr.round(2).to_dict(orient="index")
    out["attempts"] = gt.assign(tx=rm.tx.values).groupby("tx").attempts.agg(["mean", "max", lambda s: int((s > 1).sum())]).to_dict(orient="index")
    out["window_to_noise_db_min"] = float(gt.window_to_noise_db.min())
    json.dump(out, open(f"{RESULTS}/hw_diagnostics.json", "w"), indent=1)
    d.to_csv(f"{RESULTS}/hw_diagnostics_per_capture.csv", index=False)
    for k_, v in out.items():
        print(k_, v)


if __name__ == "__main__":
    main()
