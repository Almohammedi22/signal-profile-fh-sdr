"""Result figures for the paper (vector PDF, journal style).

  python figs_results.py [stage1] [snr] [cfo] [lodo]      (default: all)
Every figure is computed from the stored results or run scores, so it can be regenerated from the release.
Colors follow a fixed categorical order; every series also has its own marker so that the figures remain
readable in grayscale print.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from config import CACHE, DATA, FIGURES, HERE, RESULTS, RUNS

FIG = FIGURES
RES = RESULTS
RUNS = RUNS
C = CACHE
plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "lines.linewidth": 1.1, "lines.markersize": 3.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARK = ["o", "s", "^", "D", "v", "P"]
ARCHS = ["cnn", "cnn_lstm", "cnn_gru", "cldnn", "bilstm", "transformer"]
ARCH_LABEL = {"cnn": "CNN-only", "cnn_lstm": "CNN–LSTM", "cnn_gru": "CNN–GRU", "cldnn": "CLDNN",
              "bilstm": "BiLSTM-only", "transformer": "Transformer"}


def grid(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_stage1():
    """(a) detection probability vs SNR in non-overlapping 1-dB bins, with theory for both detectors;
    (b) false-alarm probability when the true noise power differs from the assumed one."""
    from scipy import stats as st
    sm = pd.read_csv(f"{C}/syn_meta.csv")
    E = np.load(f"{C}/syn_energy.npy").astype(np.float64)
    summ = json.load(open(f"{RES}/stage1_summary.json"))
    thr, gamma, N, K = summ["threshold"], summ["cfar_gamma"], summ["N"], summ["K_reference"]
    te = (sm.split == "test").values & (sm.y_state.values > 0)
    snr = sm.snr_db.values
    off = pd.read_csv(f"{RES}/stage1_noise_offset.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.35), gridspec_kw={"wspace": 0.32})
    lo_e = np.arange(-20, 2, 1.0)
    cen, pdm, lo, hi = [], [], [], []
    for x in lo_e:
        k = te & (snr >= x) & (snr < x + 1)
        n = int(k.sum())
        kk = int((E[k] > thr).sum())
        l, h = wilson(kk, n)
        cen.append(x + 0.5); pdm.append(kk / n); lo.append(l); hi.append(h)
    cen, pdm, lo, hi = map(np.array, (cen, pdm, lo, hi))
    g = np.linspace(-20, 2, 221)
    r = 10 ** (g / 10)
    a.plot(g, st.ncx2.sf(2 * N * thr, 2 * N, 2 * N * r), color=INK, label="Known noise (theory)")
    a.plot(g, st.ncf.sf(gamma, 2 * N, 2 * K, 2 * N * r), color=SLOTS[1], linestyle=(0, (4, 2)),
           label="Noise reference (theory)")
    a.errorbar(cen, pdm, yerr=[pdm - lo, hi - pdm], fmt="o", color=SLOTS[0], ms=3.0, elinewidth=0.6, capsize=0,
               label="Measured (1-dB bins)")
    a.set_xlabel("SNR (dB)")
    a.set_ylabel("Detection probability $P_D$")
    a.set_xlim(-20.5, 2.5)
    a.set_ylim(-0.02, 1.03)
    a.legend(loc="lower right", handlelength=2.0, borderaxespad=0.2)
    grid(a)
    a.text(-0.17, 1.02, "(a)", transform=a.transAxes, fontsize=8, fontweight="bold", va="bottom")
    b.semilogy(off.u_dB, off.pfa_fixed_theory, color=INK, label="Fixed threshold, theory")
    n_idle = int(((sm.split == "test") & (sm.y_state == 0)).sum())
    m = (off.pfa_fixed_measured * n_idle >= 5).values
    b.semilogy(off.u_dB[m][::2], off.pfa_fixed_measured[m][::2], "o", color=SLOTS[0], ms=3.0,
               label="Fixed threshold, measured")
    b.semilogy(off.u_dB, off.pfa_cfar_theory, color=SLOTS[1], linestyle=(0, (4, 2)), label="Noise reference, theory")
    b.set_xlabel("Noise power above the assumed value (dB)")
    b.set_ylabel("False-alarm probability $P_{FA}$")
    b.set_xlim(-1.0, 1.0)
    b.set_ylim(1e-4, 1.5)
    b.legend(loc="lower right", handlelength=2.2)
    grid(b)
    b.text(-0.17, 1.02, "(b)", transform=b.transAxes, fontsize=8, fontweight="bold", va="bottom")
    fig.savefig(f"{FIG}/fig_stage1.pdf")
    plt.close(fig)
    print("fig_stage1.pdf", "measured P_D by bin:", dict(zip(cen.round(1), pdm.round(3))))


def syn_scores():
    sm = pd.read_csv(f"{C}/syn_meta.csv")
    out = {}
    for f in sorted(glob.glob(f"{RUNS}/syn_*/info.json")):
        info = json.load(open(f))
        if info["ncls"] != 2:
            continue
        s = np.load(f.replace("info.json", "scores.npz"))
        rws = s["test_all_rows"]
        out.setdefault(info["arch"], []).append((info["seed"], rws, s["test_all_prob"][:, 1]))
    return sm, out


def fig_snr():
    sm, runs = syn_scores()
    edges = np.arange(-20, 25.01, 5.0)
    mids = 0.5 * (edges[:-1] + edges[1:])
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.45), sharey=True, gridspec_kw={"wspace": 0.08})
    for j, arch in enumerate(ARCHS):
        if arch not in runs:
            continue
        for ax, cls in zip(axs, [2, 1]):
            vals = []
            for seed, rws, pa in runs[arch]:
                st = sm.y_state.values[rws]
                snr = sm.snr_db.values[rws]
                r = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    kk = (st == cls) & (snr >= lo) & (snr < hi + (1e-9 if hi == edges[-1] else 0))
                    pred_adv = pa[kk] >= 0.5
                    r.append(np.mean(pred_adv) if cls == 2 else np.mean(~pred_adv))
                vals.append(r)
            v = np.array(vals)
            mu = v.mean(0)
            sd = v.std(0, ddof=1) if len(v) > 1 else np.zeros_like(mu)
            off = (j - 2.5) * 0.35
            ax.errorbar(mids + off, mu, yerr=sd, color=SLOTS[j], marker=MARK[j], ms=3.2, elinewidth=0.6,
                        capsize=0, linewidth=0.9, label=f"{ARCH_LABEL[arch]} ($n$={len(v)})")
    for ax, t in zip(axs, ["Adversarial recall $R_A$", "Friendly recall $R_F$"]):
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(edges)
        ax.set_xlim(-21, 26)
        ax.set_ylim(-0.03, 1.04)
        ax.axvspan(-21, -10, color="#f0efec", zorder=0)
        grid(ax)
        ax.set_title(t, loc="left")
    axs[0].set_ylabel("Recall (mean $\\pm$ s.d. over runs)")
    axs[0].text(-16.8, 0.52, "below the\ninformative\nrange", ha="center", va="center", fontsize=6.8, color=MUTED)
    axs[1].legend(loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.6)
    fig.savefig(f"{FIG}/fig_snr.pdf")
    plt.close(fig)
    print("fig_snr.pdf")


CFO = {"Pluto_02": -1387.4, "Pluto_04": -658.0, "Pluto_05": -2403.7}
TXL = {"Pluto_02": "Pluto\\_02", "Pluto_04": "Pluto\\_04", "Pluto_05": "Pluto\\_05"}


def fig_cfo():
    rows = []
    for f in sorted(glob.glob(f"{RUNS}/hw_*/cfo_shift.json")):
        d = os.path.dirname(f)
        info = json.load(open(f"{d}/info.json"))
        j = json.load(open(f))
        parts = info["job"].split("_")
        for key, rec in j["recall"].items():
            if key == "calibrated":
                continue
            df = float(key)
            for mod in ["PAM4", "PAM8"]:
                rows.append({"held": j["heldout"], "regime": parts[3], "aug": float(info.get("cfo_aug_khz", 0)),
                             "seed": info["seed"], "shift": df, "eff": CFO[j["heldout"]] + df, "mod": mod, "rec": rec[mod]})
    if not rows:
        print("no cfo_shift results")
        return
    t = pd.DataFrame(rows)
    helds = ["Pluto_02", "Pluto_04", "Pluto_05"]
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.25), sharey=True, gridspec_kw={"wspace": 0.08})
    series = [("real", 0.0, "Real-only", SLOTS[0], MARK[0]), ("real", 3.0, "Real-only + offset augmentation", SLOTS[2], MARK[2])]
    for ax, held in zip(axs, helds):
        tr = [CFO[x] for x in helds if x != held]
        ax.axvspan(min(tr) / 1e3, max(tr) / 1e3, color="#f0efec", zorder=0)
        ax.axvline(CFO[held] / 1e3, color=MUTED, linewidth=0.6, linestyle=(0, (1, 1.5)))
        for k, (reg, aug, lab, col, mk) in enumerate(series):
            for mod, ls in [("PAM8", "-"), ("PAM4", (0, (3, 1.5)))]:
                q = t[(t.held == held) & (t.regime == reg) & (t.aug == aug) & (t["mod"] == mod)]
                if q.empty:
                    continue
                g = q.groupby("eff").rec
                mu, sd = g.mean(), g.std(ddof=1).fillna(0)
                ax.plot(mu.index / 1e3, mu.values, color=col, linestyle=ls, marker=mk, ms=2.6,
                        markevery=2, label=f"{lab}, {mod}")
                if q.seed.nunique() > 1:
                    ax.fill_between(mu.index / 1e3, (mu - sd).values, np.minimum(mu + sd, 1).values,
                                    color=col, alpha=0.12, linewidth=0)
        ax.set_title(f"Held out: {held.replace('_', chr(95))}", loc="left")
        ax.set_xlabel("Effective offset after shift (kHz)")
        ax.set_xlim(-5.6, 2.8)
        ax.set_ylim(-0.03, 1.04)
        grid(ax)
    axs[0].set_ylabel("Recall of the adversarial label")
    h, l = axs[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2), columnspacing=1.5, handlelength=2.4)
    fig.savefig(f"{FIG}/fig_cfo.pdf")
    plt.close(fig)
    print("fig_cfo.pdf")


def fig_lodo():
    """Held-out balanced accuracy and PAM8 recall per fold and configuration; one dot per training run."""
    h = pd.read_csv(f"{RES}/paper/hw_runs.csv")
    h = h[h.ncls == 2]
    confs = [("real", 0.0, "Real-only"), ("combined", 0.0, "Combined"), ("real", 3.0, "Real-only + offset aug."),
             ("combined", 3.0, "Combined + offset aug.")]
    helds = ["Pluto_02", "Pluto_04", "Pluto_05"]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.4), gridspec_kw={"wspace": 0.25})
    for ax, met, lab in [(axs[0], "heldout_bacc", "Held-out balanced accuracy"), (axs[1], "rec_PAM8", "Held-out PAM8 recall")]:
        for i, held in enumerate(helds):
            for k, (reg, aug, cl) in enumerate(confs):
                q = h[(h.heldout == held) & (h.regime == reg) & (h.cfo_aug_khz == aug)][met].values
                if len(q) == 0:
                    continue
                x = i + (k - 1.5) * 0.18
                jit = np.linspace(-0.04, 0.04, len(q)) if len(q) > 1 else np.zeros(1)
                ax.plot(x + jit, q, linestyle="none", marker=MARK[k], ms=3.0, color=SLOTS[k], alpha=0.85,
                        markeredgewidth=0, label=cl if i == 0 else None)
                ax.plot([x - 0.07, x + 0.07], [q.mean()] * 2, color=INK, linewidth=1.0)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"Held out: {x}" for x in helds])
        ax.set_title(lab, loc="left")
        grid(ax)
    axs[0].set_ylim(0.85, 1.005)
    axs[1].set_ylim(-0.03, 1.04)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), columnspacing=1.0, handlelength=1.2)
    fig.savefig(f"{FIG}/fig_lodo.pdf")
    plt.close(fig)
    print("fig_lodo.pdf")


def pr_curve(y, sc):
    o = np.argsort(-sc, kind="mergesort")
    y = y[o]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    rec = tp / max(y.sum(), 1)
    prec = tp / np.maximum(tp + fp, 1)
    return rec, prec


def fig_pr():
    """Precision-recall curves (adversarial = positive), predictions pooled over the training runs."""
    sm, runs = syn_scores()
    rm = pd.read_csv(f"{C}/real_meta.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.5), gridspec_kw={"wspace": 0.25})
    ax = axs[0]
    for j, arch in enumerate(ARCHS):
        if arch not in runs:
            continue
        ys, ss = [], []
        for seed, rws, pa in runs[arch]:
            st = sm.y_state.values[rws]
            snr = sm.snr_db.values[rws]
            k = (st > 0) & (snr >= -10)
            ys.append((st[k] == 2).astype(int))
            ss.append(pa[k])
        r, p = pr_curve(np.concatenate(ys), np.concatenate(ss))
        ax.plot(r, p, color=SLOTS[j], label=f"{ARCH_LABEL[arch]}", linewidth=1.0)
    ax.set_xlabel("Adversarial recall")
    ax.set_ylabel("Adversarial precision")
    ax.set_title("(a) Synthetic test split, SNR $\\geq -10$ dB", loc="left")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.5, 1.01)
    ax.legend(loc="lower left", ncol=2, columnspacing=0.8, handlelength=1.5)
    grid(ax)
    ax = axs[1]
    confs = [("real", 0.0, "Real-only"), ("combined", 0.0, "Combined"), ("real", 3.0, "Real-only + offset aug."),
             ("combined", 3.0, "Combined + offset aug.")]
    for k, (reg, aug, lab) in enumerate(confs):
        ys, ss = [], []
        for f in sorted(glob.glob(f"{RUNS}/hw_Pluto_05_{reg}_s*/info.json")):
            info = json.load(open(f))
            if info["ncls"] != 2 or float(info.get("cfo_aug_khz", 0)) != aug:
                continue
            sc = np.load(f.replace("info.json", "scores.npz"))
            rws = sc["heldout_rows"]
            ys.append((rm.cls.values[rws] == "Adversarial").astype(int))
            ss.append(sc["heldout_prob"][:, 1])
        if not ys:
            continue
        r, p = pr_curve(np.concatenate(ys), np.concatenate(ss))
        ax.plot(r, p, color=SLOTS[k], label=f"{lab} ($n$={len(ys)})", linewidth=1.0,
                linestyle="-" if aug == 0 else (0, (4, 1.5)))
    ax.set_xlabel("Adversarial recall")
    ax.set_title("(b) Hardware, held-out Pluto_05", loc="left")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.5, 1.01)
    ax.legend(loc="lower left", handlelength=2.0)
    grid(ax)
    fig.savefig(f"{FIG}/fig_pr.pdf")
    plt.close(fig)
    print("fig_pr.pdf")


if __name__ == "__main__":
    want = sys.argv[1:] or ["stage1", "snr", "cfo", "lodo", "pr"]
    for w in want:
        {"stage1": fig_stage1, "snr": fig_snr, "cfo": fig_cfo, "lodo": fig_lodo, "pr": fig_pr}[w]()
