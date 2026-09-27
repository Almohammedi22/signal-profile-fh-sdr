"""Carrier-frequency offset of each PlutoSDR transmitter relative to the receiver (no new captures).

Uses the captures with real-valued constellations (BPSK, ASK, OOK, PAM4, PAM8). Squaring an
observation removes the modulation and leaves a spectral line at twice the total frequency offset.
Observations are 256 samples at 62.5 kS/s (1 MS/s averaged 16:1). BPSK is sent at baseband; the
amplitude-keyed profiles are sent on known digital subcarriers (55, 45, 40 and 35 kHz), which alias at
62.5 kS/s and are removed before the transmitter offset is reported. The script also compares the
first and last quarters of the capture order to check for drift over the collection.
Usage: python cfo_offsets.py [path/to/real_sdr_dataset_31800.mat]
"""
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

FS_OBS, NFFT = 62_500.0, 4096
SUBCARRIER_HZ = {"BPSK": 0.0, "ASK": 55e3, "OOK": 45e3, "PAM4": 40e3, "PAM8": 35e3}


def alias(f):
    """Frequency as seen at the observation rate, in [-FS/2, FS/2)."""
    return (f + FS_OBS / 2) % FS_OBS - FS_OBS / 2


def main(path):
    d = sio.loadmat(path, squeeze_me=True)
    iq, mod = d["realIQ"], np.array(d["modulationName"], dtype=str)
    tx, order = np.array(d["txUnit"], dtype=str), np.asarray(d["captureOrder"])
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / FS_OBS))
    idx = np.where(np.isin(mod, list(SUBCARRIER_HZ)))[0]
    line = np.empty(len(idx))
    for b in range(0, len(idx), 400):                       # batches keep memory small
        ii = idx[b:b + 400]
        x = iq[ii, :, 0].astype(np.complex64) + 1j * iq[ii, :, 1].astype(np.complex64)
        x -= x.mean(axis=1, keepdims=True)
        s = np.abs(np.fft.fftshift(np.fft.fft(x ** 2, NFFT, axis=1), axes=1))
        k = s.argmax(axis=1)
        r = np.arange(len(k))
        a, c0, c = (np.log(s[r, np.clip(k + o, 0, NFFT - 1)] + 1e-9) for o in (-1, 0, 1))
        line[b:b + 400] = freqs[k] + 0.5 * (a - c) / (a - 2 * c0 + c + 1e-12) * FS_OBS / NFFT
    # the squared line sits at alias(2 * (f_sub + offset)); undo the aliasing and the known subcarrier,
    # keeping the solution with the smallest magnitude (|offset| < FS/4)
    sub = np.array([alias(SUBCARRIER_HZ[m]) for m in mod[idx]])
    cand = np.stack([(line + n * FS_OBS) / 2 - sub for n in range(-3, 4)])
    offset = cand[np.argmin(np.abs(cand), axis=0), np.arange(len(idx))]
    df = pd.DataFrame({"tx": tx[idx], "modulation": mod[idx], "capture_order": order[idx], "offset_hz": offset})
    df.to_csv("cfo_estimates.csv", index=False)

    print("Offset relative to the receiver, all five modulations after subcarrier removal:")
    for t, g in df.groupby("tx"):
        q1, q3 = g.offset_hz.quantile([0.25, 0.75])
        print(f"  {t}: median {g.offset_hz.median() / 1e3:+.2f} kHz, interquartile range {q3 - q1:.1f} Hz")
    print("\nMedian offset (Hz) by modulation after subcarrier removal:")
    print(df.pivot_table(index="modulation", columns="tx", values="offset_hz", aggfunc="median").round(1))
    print("\nDrift, median offset in the first and last quarters of the capture order:")
    for t, g in df.groupby("tx"):
        lo, hi = g.capture_order.quantile([0.25, 0.75])
        first, last = g[g.capture_order <= lo].offset_hz.median(), g[g.capture_order >= hi].offset_hz.median()
        print(f"  {t}: {first:.1f} Hz -> {last:.1f} Hz (change {abs(last - first):.1f} Hz)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "results_real_dataset_31800/dataset_31800_current/real_sdr_dataset_31800.mat")
