"""Round-2 analyses on stored predictions (Reviewer 2: B2 intervals, C5 confusions, C6 calibration, C9 retries)."""
import json, sys
import numpy as np, pandas as pd
import analyze_results as A
from config import DATA, RESULTS
out = {}
def ece(conf, correct, nb=15):
    b = np.minimum((conf * nb).astype(int), nb - 1); e = 0.0
    for k in range(nb):
        s = b == k
        if s.any(): e += s.mean() * abs(correct[s].mean() - conf[s].mean())
    return float(e)
def calib(pa, y):
    pred = pa >= 0.5; conf = np.where(pred, pa, 1 - pa); cor = pred == (y == 1)
    band = (conf >= 0.85) & (conf < 0.95)
    return {"ece": ece(conf, cor), "acc_conf_085_095": float(cor[band].mean()) if band.any() else None,
            "mean_conf_085_095": float(conf[band].mean()) if band.any() else None,
            "acc_conf_ge_09": float(cor[conf >= 0.9].mean()), "mean_conf_ge_09": float(conf[conf >= 0.9].mean())}
syn, cal_syn = [], []
for info, s, d in A.load_runs("syn_"):
    rws = s["test_all_rows"]; st = A.sm.y_state.values[rws]; snr = A.sm.snr_db.values[rws]
    pa = A.padv_of(info, s["test_all_prob"]); occ = (st > 0) & (snr >= -10)
    y = (st[occ] == 2).astype(int); p = pa[occ]; m = A.metrics(y, p)
    nF, nA = int((y == 0).sum()), int((y == 1).sum())
    hw = 1.96 * 0.5 * np.sqrt(m["rec_F"] * (1 - m["rec_F"]) / nF + m["rec_A"] * (1 - m["rec_A"]) / nA)
    syn.append({"job": info["job"], "arch": info["arch"], "ncls": info["ncls"], "bacc": m["bacc"], "half_width": hw, "nF": nF, "nA": nA})
    if info["arch"] == "cnn_lstm": cal_syn.append(calib(p, y) | {"job": info["job"]})
out["syn_intervals"] = syn; out["calib_syn_cnn_lstm"] = cal_syn
hwc = []
for info, s, d in A.load_runs("hw_"):
    if info["ncls"] != 2 or "combined" in info["job"]: continue
    rws = s["heldout_rows"]; y = (A.rm.cls.values[rws] == "Adversarial").astype(int); p = A.padv_of(info, s["heldout_prob"])
    m = A.metrics(y, p); nF, nA = int((y == 0).sum()), int((y == 1).sum())
    hwc.append(calib(p, y) | {"job": info["job"], "bacc": m["bacc"],
               "half_width": 1.96 * 0.5 * np.sqrt(m["rec_F"] * (1 - m["rec_F"]) / nF + m["rec_A"] * (1 - m["rec_A"]) / nA)})
out["calib_hw"] = hwc
# C5: friendly windows labelled adversarial above 0 dB (binary CNN-LSTM), and the class the 20-class net gives them
fa_mods = {}; conf20 = {}
for info, s, d in A.load_runs("syn_"):
    rws = s["test_all_rows"]; st = A.sm.y_state.values[rws]; snr = A.sm.snr_db.values[rws]; mods = A.sm["mod"].values[rws]
    sel = (st == 1) & (snr >= 0)
    if info["arch"] == "cnn_lstm" and info["ncls"] == 2:
        err = A.padv_of(info, s["test_all_prob"])[sel] >= 0.5
        for mm in np.unique(mods[sel]): fa_mods.setdefault(mm, []).append(float(err[mods[sel] == mm].mean()))
    if info["ncls"] == 20:
        pr = s["test_all_prob"][sel]; lab = np.array(info["label_order"])[pr.argmax(1)]
        wrong_adv = np.isin(lab, list(A.ADV))
        for src, dst in zip(mods[sel][wrong_adv], lab[wrong_adv]): conf20[f"{src}->{dst}"] = conf20.get(f"{src}->{dst}", 0) + 1
        out["syn20_friendly_hi_to_adv_rate"] = float(wrong_adv.mean())
out["binary_fa_rate_by_friendly_mod_hi"] = {k: float(np.mean(v)) for k, v in sorted(fa_mods.items(), key=lambda kv: -np.mean(kv[1]))}
out["syn20_confusions_friendly_to_adv_hi"] = dict(sorted(conf20.items(), key=lambda kv: -kv[1]))
# C9: repeated captures
gt = pd.read_csv(f"{DATA}/ground_truth_31800.csv", low_memory=False)
rep = gt[gt.attempts > 1]
out["retries_total"] = int(len(rep)); out["retries_by_tx"] = rep.tx_unit.value_counts().to_dict()
out["retries_by_mod"] = rep.modulation_name.value_counts().to_dict()
out["retry_rate_by_mod"] = (gt.groupby("modulation_name").attempts.apply(lambda a: float((a > 1).mean()))).sort_values(ascending=False).round(4).to_dict()
out["attempts_max"] = int(gt.attempts.max())
json.dump(out, open(f"{RESULTS}/paper/round2_checks.json", "w"), indent=1, default=float)
df = pd.DataFrame(syn); print(df.groupby(["arch", "ncls"])[["bacc", "half_width"]].mean().round(4))
print("calib syn:", [{k: round(v, 3) if isinstance(v, float) else v for k, v in c.items()} for c in cal_syn])
print("calib hw:", [(c["job"], round(c["ece"], 4), round(c["acc_conf_ge_09"], 4), round(c["mean_conf_ge_09"], 4)) for c in hwc])
print("FA by friendly mod (hi SNR):", {k: round(v, 3) for k, v in list(out["binary_fa_rate_by_friendly_mod_hi"].items())[:8]})
print("20-class friendly->adv:", list(out["syn20_confusions_friendly_to_adv_hi"].items())[:10], out.get("syn20_friendly_hi_to_adv_rate"))
print("retries:", out["retries_total"], out["retries_by_tx"], list(out["retry_rate_by_mod"].items())[:8], "max attempts", out["attempts_max"])
