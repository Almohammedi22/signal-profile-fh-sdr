"""Build float32 caches of the synthetic and hardware datasets (run once)."""
import sys, json
import numpy as np, pandas as pd, scipy.io as sio
from mat73 import Mat73
from config import CACHE, DATA, FIGURES, HERE, RESULTS, RUNS

D, C = DATA, CACHE
# ---------------- hardware: v5 MAT file + clean-split CSV, cross-checked against the v7.3 clean-split file
r = sio.loadmat(f"{D}/real_sdr_dataset_31800.mat", squeeze_me=True)
X = r["realIQ"].astype(np.float32)
meta = pd.DataFrame({"modulation": np.array(r["modulationName"], str), "cls": np.array(r["className"], str),
                     "tx": np.array(r["txUnit"], str), "orig_split": np.array(r["split"], str),
                     "session": np.asarray(r["sessionID"]).astype(int), "order": np.asarray(r["captureOrder"]).astype(int)})
cs = pd.read_csv(f"{D}/clean_split_full_metadata.csv").sort_values("DatasetRow")
assert (cs.DatasetRow.values == np.arange(1, len(X) + 1)).all()
for a, b in [("tx", "TxUnit"), ("modulation", "ModulationName"), ("cls", "ClassName"), ("order", "CaptureOrder"),
             ("orig_split", "OriginalSplit")]:
    assert (meta[a].values == cs[b].values).all(), a
meta["split"] = cs.CleanSplit.values
meta["leak_group"] = cs.LeakageComponentID.values
import os
v73 = f"{D}/real_sdr_dataset_31800_clean_split.mat"
if os.path.exists(v73):                      # optional cross-check against the MATLAB v7.3 clean-split file
    X73 = Mat73(v73).read("realIQ")
    print("clean-split v7.3 realIQ shape", X73.shape, "max |diff| vs v5 file:", float(np.abs(X73.astype(np.float32) - X).max()))
np.save(f"{C}/real_X.npy", X)
meta.to_csv(f"{C}/real_meta.csv", index=False)
print(meta.groupby(["tx", "split"]).size().unstack())
# ---------------- synthetic Dataset v5
m = Mat73(f"{D}/dataset_v5.mat")
Xs = m.read("X").astype(np.float32)
smeta = pd.DataFrame({k: m.read(k).ravel() for k in ["y_state", "snr_db", "y_system", "y_fh_pattern", "channel_id",
                                                      "collision_flag", "intruder_pattern_id", "intruder_proximity_flag",
                                                      "time_slot", "episode_id", "tx_id", "rx_id"]})
split = np.full(len(Xs), "", object)
for nm in ["train_idx", "val_idx", "test_idx"]:
    split[m.read(nm).ravel().astype(int) - 1] = nm.split("_")[0]
smeta["split"] = split
names = {k: m.strings(k) for k in ["systemNames", "fhPatternNames", "stateNames", "friendlyModNames",
                                   "adversarialModNames", "intruderPatternNames"]}
json.dump(names, open(f"{C}/syn_names.json", "w"), indent=1)
np.save(f"{C}/syn_X.npy", Xs)
np.save(f"{C}/syn_energy.npy", (Xs.astype(np.float64) ** 2).sum(2).mean(1).astype(np.float32))   # Stage 1 statistic, Eq. (11)
np.save(f"{C}/syn_channel_history.npy", m.read("channel_history").astype(np.int16))
smeta.to_csv(f"{C}/syn_meta.csv", index=False)
print(Xs.shape, smeta.split.value_counts().to_dict())
print(json.dumps(names)[:600])
