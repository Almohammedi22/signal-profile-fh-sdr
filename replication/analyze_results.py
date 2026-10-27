"""Analysis of the repeated-run study reported in the paper (works on whatever runs have finished).

Outputs in {RESULTS}/paper/:
  syn_runs.csv      one row per synthetic run: occupied-window metrics (all, informative SNR >= -10 dB,
                    SNR bins), AUPRC, adversarial recall at a validation-set threshold for P_F->A = 1 %
  syn_summary.csv   mean, sd and 95 % t half-width over seeds per (arch, ncls)
  e2e_runs.csv      three-state confusion of the complete pipeline per run (Stage 1 threshold 1.108),
                    at tau = 0.5 and at cost-based thresholds tau = 1/(1 + delta/gamma)
  e2e_snr.csv       per-run, per-SNR-bin end-to-end outcome of adversarial and friendly windows
  abstain.csv       energy-based and confidence-based abstention for low-SNR detections
  hw_runs.csv       one row per hardware run: held-out and in-domain metrics, recall per waveform,
                    calibrated frequency-shift result (if cfo_shift.json exists)
  hw_summary.csv    mean, sd, t half-width per (ncls, cfo_aug, regime, heldout)
  hw_paired.csv     paired seed-wise differences between regimes and augmentation settings
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
from config import CACHE, DATA, FIGURES, HERE, RESULTS, RUNS

RUNS, OUT = RUNS, f"{RESULTS}/paper"
C = CACHE
rm = pd.read_csv(f"{C}/real_meta.csv")
sm = pd.read_csv(f"{C}/syn_meta.csv")
NAMES = json.load(open(f"{C}/syn_names.json"))
SYS = NAMES["systemNames"]
ADV = set(NAMES["adversarialModNames"])
sm["mod"] = [SYS[int(s) - 1] if s >= 1 else "Idle" for s in sm.y_system]
E = np.load(f"{C}/syn_energy.npy").astype(np.float64)
THR = 1.108152775301702
BINS = [(-20, -10), (-10, 0), (0, 10), (10, 25.01)]
BIN_NAMES = ["[-20,-10)", "[-10,0)", "[0,10)", "[10,25]"]
TAUS = {"tau0.5": 0.5, "tau0.2": 0.2, "tau0.1": 0.1, "tau0.8": 0.8}      # delta/gamma = 1, 4, 9, 1/4


def padv_of(info, prob):
    if info["ncls"] == 2:
        return prob[:, 1]
    idx = [i for i, m in enumerate(info["label_order"]) if m in ADV]
    return prob[:, idx].sum(1)


def metrics(y, pa, thr=0.5):
    pred = pa >= thr
    rf = float(np.mean(~pred[y == 0])) if (y == 0).any() else np.nan
    ra = float(np.mean(pred[y == 1])) if (y == 1).any() else np.nan
    return {"acc": float(np.mean(pred == (y == 1))), "rec_F": rf, "rec_A": ra, "bacc": 0.5 * (rf + ra)}


def auprc(y, s):
    """Average precision with the adversarial class as positive (step-wise interpolation)."""
    o = np.argsort(-s, kind="mergesort")
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float(np.sum(prec * y) / max(y.sum(), 1))


def tci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def summarize(df, keys, cols):
    g = df.groupby(keys)
    out = g[cols].mean().add_suffix("_mean")
    out = out.join(g[cols].std(ddof=1).add_suffix("_sd"))
    out = out.join(g[cols].agg(tci).add_suffix("_ci"))
    out["n_runs"] = g.size()
    return out.reset_index()


def load_runs(prefix):
    for f in sorted(glob.glob(f"{RUNS}/{prefix}*/info.json")):
        info = json.load(open(f))
        yield info, np.load(f.replace("info.json", "scores.npz")), os.path.dirname(f)


# ----------------------------------------------------------------------------------------- synthetic
def syn_analysis():
    rows, e2e, e2e_snr, abst = [], [], [], []
    for info, s, d in load_runs("syn_"):
        rws = s["test_all_rows"]
        st = sm.y_state.values[rws]
        snr = sm.snr_db.values[rws]
        pa_all = padv_of(info, s["test_all_prob"])
        occ = st > 0
        y = (st[occ] == 2).astype(int)
        pa = pa_all[occ]
        so = snr[occ]
        r = {"job": info["job"], "arch": info["arch"], "seed": info["seed"], "ncls": info["ncls"],
             "epochs": info["epochs"], "n_params": info["n_params"], "train_s": info.get("seconds", np.nan)}
        r.update({f"all_{k}": v for k, v in metrics(y, pa).items()})
        inf = so >= -10
        r.update({f"inf_{k}": v for k, v in metrics(y[inf], pa[inf]).items()})
        for (lo, hi), nm in zip(BINS, BIN_NAMES):
            b = (so >= lo) & (so < hi)
            mm = metrics(y[b], pa[b])
            r[f"{nm}_acc"], r[f"{nm}_rec_F"], r[f"{nm}_rec_A"], r[f"{nm}_n"] = mm["acc"], mm["rec_F"], mm["rec_A"], int(b.sum())
        r["all_auprc"] = auprc(y, pa)
        r["inf_auprc"] = auprc(y[inf], pa[inf])
        # threshold for P_F->A = 1 % set on the validation occupied windows
        vr = s["val_all_rows"]
        vst = sm.y_state.values[vr]
        vpa = padv_of(info, s["val_all_prob"])
        vf = vpa[vst == 1]
        t1 = float(np.quantile(vf, 0.99))
        r["thr_fa1"] = t1
        m1 = metrics(y, pa, t1)
        r["fa1_rec_A_all"], r["fa1_pFA_all"] = m1["rec_A"], 1 - m1["rec_F"]
        m1i = metrics(y[inf], pa[inf], t1)
        r["fa1_rec_A_inf"], r["fa1_pFA_inf"] = m1i["rec_A"], 1 - m1i["rec_F"]
        # per-waveform accuracy above 10 dB (which waveform types carry the high-SNR errors)
        mods = sm["mod"].values[rws][occ]
        hi = so >= 10
        for mm in np.unique(mods):
            k = hi & (mods == mm)
            r[f"hi_{mm}"] = float(np.mean((pa[k] >= 0.5) == (y[k] == 1)))
        r["hi_rec_F"] = float(np.mean(pa[hi & (y == 0)] < 0.5))
        r["hi_rec_A"] = float(np.mean(pa[hi & (y == 1)] >= 0.5))
        # classifier output on idle windows (shows what Stage 2 would do with false alarms)
        r["idle_frac_adv"] = float(np.mean(pa_all[st == 0] >= 0.5))
        if info["ncls"] == 20:
            lab = np.array([info["label_order"].index(mm) for mm in mods])
            pred20 = s["test_all_prob"][occ].argmax(1)
            r["acc20"] = float(np.mean(pred20 == lab))
            r["acc20_inf"] = float(np.mean((pred20 == lab)[inf]))
            adv_idx = {i for i, mm in enumerate(info["label_order"]) if mm in ADV}
            same_profile = np.array([(a in adv_idx) == (b in adv_idx) for a, b in zip(pred20, lab)])
            r["frac_err20_within_profile_inf"] = float(np.mean(same_profile[inf & (pred20 != lab)])) if (inf & (pred20 != lab)).any() else 0.0
        rows.append(r)
        if info["ncls"] != 2:
            continue
        # ---------------- end-to-end three-state pipeline
        det = E[rws] > THR
        taus = dict(TAUS)
        taus["taufa1"] = t1                      # validation threshold for P_F->A = 1 %
        for tn, tau in taus.items():
            out = np.where(~det, 0, np.where(pa_all >= tau, 2, 1))
            cm = pd.crosstab(pd.Series(st, name="true"), pd.Series(out, name="pred")).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
            rec = np.diag(cm.values) / cm.values.sum(1)
            e = {"job": info["job"], "arch": info["arch"], "seed": info["seed"], "tau": tn,
                 "acc": float(np.trace(cm.values) / cm.values.sum()), "bacc3": float(rec.mean()),
                 "rec_I": rec[0], "rec_F": rec[1], "rec_A": rec[2]}
            for i, a in enumerate("IFA"):
                for j, b in enumerate("IFA"):
                    e[f"n_{a}{b}"] = int(cm.values[i, j])
            # informative range
            k = snr >= -10
            occk = k & (st > 0)
            e["inf_rec_F"] = float(np.mean(out[occk & (st == 1)] == 1))
            e["inf_rec_A"] = float(np.mean(out[occk & (st == 2)] == 2))
            e2e.append(e)
            if tn == "tau0.5":
                for (lo, hi), nm in zip(BINS, BIN_NAMES):
                    b = (snr >= lo) & (snr < hi)
                    for cls, cn in [(1, "F"), (2, "A")]:
                        kk = b & (st == cls)
                        e2e_snr.append({"job": info["job"], "arch": info["arch"], "seed": info["seed"], "bin": nm,
                                        "true": cn, "n": int(kk.sum()), "to_idle": float(np.mean(out[kk] == 0)),
                                        "to_F": float(np.mean(out[kk] == 1)), "to_A": float(np.mean(out[kk] == 2))})
        # ---------------- abstention for low-energy detections: E below 1 + 10^(s/10) -> "uncertain"
        for s_db in [-5.0, -2.5, 0.0]:
            lam_u = 1 + 10 ** (s_db / 10)
            unc = det & (E[rws] <= lam_u)
            acc_mask = det & ~unc
            out = np.where(~det, 0, np.where(pa_all >= 0.5, 2, 1))
            a = {"job": info["job"], "arch": info["arch"], "seed": info["seed"], "rule": f"energy_snr<{s_db:g}dB",
                 "frac_detections_flagged": float(unc.sum() / det.sum())}
            for cls, cn in [(0, "I"), (1, "F"), (2, "A")]:
                kk = st == cls
                a[f"{cn}_flagged"] = float(np.mean(unc[kk]))
                a[f"{cn}_correct_unflagged"] = float(np.mean((out[kk] == cls) & ~unc[kk]))
            a["acc_on_accepted_occupied"] = float(np.mean(out[acc_mask & (st > 0)] == st[acc_mask & (st > 0)]))
            abst.append(a)
        for c in [0.8, 0.9, 0.95]:
            conf = np.maximum(pa_all, 1 - pa_all)
            unc = det & (conf < c)
            out = np.where(~det, 0, np.where(pa_all >= 0.5, 2, 1))
            a = {"job": info["job"], "arch": info["arch"], "seed": info["seed"], "rule": f"confidence<{c:g}",
                 "frac_detections_flagged": float(unc.sum() / det.sum())}
            for cls, cn in [(0, "I"), (1, "F"), (2, "A")]:
                kk = st == cls
                a[f"{cn}_flagged"] = float(np.mean(unc[kk]))
                a[f"{cn}_correct_unflagged"] = float(np.mean((out[kk] == cls) & ~unc[kk]))
            acc_mask = det & ~unc
            a["acc_on_accepted_occupied"] = float(np.mean(out[acc_mask & (st > 0)] == st[acc_mask & (st > 0)]))
            abst.append(a)
    return pd.DataFrame(rows), pd.DataFrame(e2e), pd.DataFrame(e2e_snr), pd.DataFrame(abst)


# ----------------------------------------------------------------------------------------- hardware
def hw_analysis():
    rows = []
    mods_all = sorted(rm.modulation.unique())
    for info, s, d in load_runs("hw_"):
        parts = info["job"].split("_")
        r = {"job": info["job"], "heldout": f"{parts[1]}_{parts[2]}", "regime": parts[3], "seed": info["seed"],
             "ncls": info["ncls"], "cfo_aug_khz": float(info.get("cfo_aug_khz", 0.0)), "epochs": info["epochs"],
             "best_val_loss": info["best_val_loss"], "train_s": info.get("seconds", np.nan)}
        for k in ["heldout", "indomain_test"]:
            rws = s[f"{k}_rows"]
            y = (rm.cls.values[rws] == "Adversarial").astype(int)
            pa = padv_of(info, s[f"{k}_prob"])
            r.update({f"{k}_{a}": b for a, b in metrics(y, pa).items()})
            if k == "heldout":
                r["heldout_auprc"] = auprc(y, pa)
                mods = rm.modulation.values[rws]
                for mm in mods_all:
                    sel = mods == mm
                    r[f"rec_{mm}"] = float(np.mean((pa[sel] >= 0.5) == (y[sel] == 1)))
                if info["ncls"] == 20:
                    lab = np.array([info["label_order"].index(m.replace("-", "_")) for m in mods])
                    pred = s[f"{k}_prob"].argmax(1)
                    r["acc20"] = float(np.mean(pred == lab))
                    ofdm = [info["label_order"].index(x) for x in ["OFDM", "OFDM_QPSK"]]
                    r["acc20_merge_ofdm"] = float(np.mean(np.isin(pred, ofdm) & np.isin(lab, ofdm) | (pred == lab)))
        cf = f"{d}/cfo_shift.json"
        if os.path.exists(cf):
            j = json.load(open(cf))
            r["calib_shift_hz"] = j["calibrated_shift_hz"]
            for mm in ["PAM4", "PAM8"]:
                r[f"calib_rec_{mm}"] = j["recall"]["calibrated"][mm]
        rows.append(r)
    return pd.DataFrame(rows)


def paired(h):
    """Seed-wise paired differences: combined - real, and augmented - not augmented."""
    out = []
    b = h[h.ncls == 2].set_index(["heldout", "regime", "cfo_aug_khz", "seed"])
    for held in sorted(h.heldout.unique()):
        for name, a_key, b_key in [("combined-real", ("combined", 0.0), ("real", 0.0)),
                                   ("real+aug - real", ("real", 3.0), ("real", 0.0)),
                                   ("combined+aug - combined", ("combined", 3.0), ("combined", 0.0)),
                                   ("real+aug - combined", ("real", 3.0), ("combined", 0.0))]:
            for met in ["heldout_bacc", "heldout_rec_A", "heldout_acc"]:
                diffs = []
                for sd in range(1, 6):
                    try:
                        diffs.append(b.loc[(held, *a_key, sd), met] - b.loc[(held, *b_key, sd), met])
                    except KeyError:
                        pass
                if diffs:
                    out.append({"heldout": held, "contrast": name, "metric": met, "n": len(diffs),
                                "mean_diff": float(np.mean(diffs)), "ci": tci(diffs),
                                "p_t": float(stats.ttest_1samp(diffs, 0).pvalue) if len(diffs) > 1 else np.nan})
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    s, e, es, ab = syn_analysis()
    if len(s):
        s.to_csv(f"{OUT}/syn_runs.csv", index=False)
        cols = [c for c in s.columns if c.startswith(("all_", "inf_", "fa1_", "[", "hi_", "acc20", "frac_err20")) or c in ("epochs", "idle_frac_adv", "train_s")]
        summarize(s, ["ncls", "arch"], cols).to_csv(f"{OUT}/syn_summary.csv", index=False)
        print(summarize(s, ["ncls", "arch"], ["all_bacc", "inf_bacc", "inf_rec_A", "inf_auprc", "fa1_rec_A_inf"])
              .round(4).to_string(index=False))
    if len(e):
        e.to_csv(f"{OUT}/e2e_runs.csv", index=False)
        es.to_csv(f"{OUT}/e2e_snr.csv", index=False)
        ab.to_csv(f"{OUT}/abstain.csv", index=False)
        print(summarize(e, ["arch", "tau"], ["acc", "bacc3", "rec_I", "rec_F", "rec_A", "n_FA", "n_AF", "n_AI"])
              .round(4).to_string(index=False))
    h = hw_analysis()
    if len(h):
        h.to_csv(f"{OUT}/hw_runs.csv", index=False)
        cols = [c for c in h.columns if c.startswith(("heldout_", "indomain_", "rec_", "calib_", "acc20"))] + ["epochs", "train_s"]
        hs = summarize(h, ["ncls", "cfo_aug_khz", "regime", "heldout"], cols)
        hs.to_csv(f"{OUT}/hw_summary.csv", index=False)
        print(hs[["ncls", "cfo_aug_khz", "regime", "heldout", "n_runs", "heldout_bacc_mean", "heldout_bacc_sd",
                  "heldout_rec_A_mean", "rec_PAM4_mean", "rec_PAM8_mean", "indomain_test_bacc_mean"]].round(4).to_string(index=False))
        p = paired(h)
        if len(p):
            p.to_csv(f"{OUT}/hw_paired.csv", index=False)
            print(p[p.metric == "heldout_bacc"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
