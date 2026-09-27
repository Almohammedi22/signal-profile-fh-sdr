#!/usr/bin/env python3
"""Collect 31,800 physically received I/Q captures using four PlutoSDRs.

Hardware topology:
  Pluto_02, Pluto_04, Pluto_05: transmitters, one active at a time
  Pluto_03: receiver

The transmitted signals are conventional DSP waveforms. The dataset stores
only I/Q samples physically received by Pluto_03. This script supports safe
resume, retries failed rows, and separates train/validation/test by repetition
session.
"""
import argparse
import csv
import json
import os
import sys
import time
import signal
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.signal import chirp

MODULATIONS = [
    ("BPSK", "Friendly", 1), ("QPSK", "Friendly", 1),
    ("8PSK", "Friendly", 1), ("16QAM", "Friendly", 1),
    ("64QAM", "Friendly", 1), ("GFSK", "Friendly", 1),
    ("MSK", "Friendly", 1), ("CPFSK", "Friendly", 1),
    ("DSSS", "Friendly", 1), ("OFDM", "Friendly", 1),
    ("OFDM_QPSK", "Friendly", 1), ("OFDM_16QAM", "Friendly", 1),
    ("ASK", "Adversarial", 2), ("OOK", "Adversarial", 2),
    ("PAM4", "Adversarial", 2), ("PAM8", "Adversarial", 2),
    ("FHSS", "Adversarial", 2), ("Radar_Pulse", "Adversarial", 2),
    ("Burst_Signal", "Adversarial", 2), ("LoRa_like", "Adversarial", 2),
]
TX_UNITS = [
    ("Pluto_02", "ip:192.168.2.1"),
    ("Pluto_04", "ip:192.168.4.1"),
    ("Pluto_05", "ip:192.168.5.1"),
]
RX_NAME = "Pluto_03"
RX_URI = "ip:192.168.3.1"

TARGET_CAPTURES = 31_800
CAPTURES_PER_COMBINATION = 530
TRAIN_PER_COMBINATION = 350
VALIDATION_PER_COMBINATION = 90
TEST_PER_COMBINATION = 90

FS = 1_000_000
FC = 915_000_000
RAW_LEN = 256
UPSAMPLE = 16
ACTIVE_LEN = RAW_LEN * UPSAMPLE
GUARD = 4096
TX_GAIN_DB = -10.0
RX_GAIN_DB = 40.0
TX_DIGITAL_RMS = 5200.0
CAPTURE_LEN = 3 * (ACTIVE_LEN + 2 * GUARD)
DETECTION_MARGIN = 2.5
MAX_RETRIES = 5
SETTLE_SECONDS = 5.0
RX_FLUSH_BUFFERS_AFTER_TX = 3
MASTER_SEED = 31800
CHECKPOINT_INTERVAL = 1
STOP_FILE_NAME = "STOP_AFTER_CURRENT_CAPTURE.txt"
STOP_REQUESTED = False

def normalize_rms(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64).reshape(-1)
    x = x - np.mean(x)
    p = float(np.mean(np.abs(x) ** 2))
    if not np.isfinite(p) or p <= 1e-12:
        raise RuntimeError("Generated waveform has zero or invalid power")
    return (x / np.sqrt(p)).astype(np.complex64)


def psk(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    symbols = rng.integers(0, m, n)
    phase = 2 * np.pi * symbols / m
    return np.exp(1j * phase).astype(np.complex64)


def qam(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    side = int(np.sqrt(m))
    levels = np.arange(-(side - 1), side, 2, dtype=np.float32)
    i = rng.choice(levels, n)
    q = rng.choice(levels, n)
    return (i + 1j * q).astype(np.complex64)


def cpfsk(n: int, rng: np.random.Generator, h: float, gaussian: bool) -> np.ndarray:
    bits = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), n)
    if gaussian:
        kernel_x = np.arange(-6, 7, dtype=np.float32)
        kernel = np.exp(-0.5 * (kernel_x / 2.2) ** 2)
        kernel /= np.sum(kernel)
        bits = np.convolve(bits, kernel, mode="same")
    phase = np.cumsum(np.pi * h * bits)
    return np.exp(1j * phase).astype(np.complex64)


def ofdm_256(rng: np.random.Generator, order: int) -> np.ndarray:
    nfft, cp = 64, 16
    blocks = []
    while sum(len(b) for b in blocks) < RAW_LEN:
        spectrum = np.zeros(nfft, dtype=np.complex64)
        carriers = np.r_[np.arange(6, 31), np.arange(34, 59)]
        data = psk(4, len(carriers), rng) if order == 4 else qam(16, len(carriers), rng)
        spectrum[carriers] = data
        td = np.fft.ifft(np.fft.ifftshift(spectrum)).astype(np.complex64)
        blocks.append(np.r_[td[-cp:], td])
    return np.concatenate(blocks)[:RAW_LEN].astype(np.complex64)


def generate_baseband(name: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = RAW_LEN
    t = np.arange(n, dtype=np.float32) / FS * UPSAMPLE

    if name == "BPSK":
        x = psk(2, n, rng)
    elif name == "QPSK":
        x = psk(4, n, rng)
    elif name == "8PSK":
        x = psk(8, n, rng)
    elif name == "16QAM":
        x = qam(16, n, rng)
    elif name == "64QAM":
        x = qam(64, n, rng)
    elif name == "GFSK":
        x = cpfsk(n, rng, h=0.32, gaussian=True)
    elif name == "MSK":
        x = cpfsk(n, rng, h=0.50, gaussian=False)
    elif name == "CPFSK":
        x = cpfsk(n, rng, h=0.37, gaussian=False)
    elif name == "DSSS":
        bits = rng.choice(np.array([-1.0, 1.0]), int(np.ceil(n / 11)))
        pn = rng.choice(np.array([-1.0, 1.0]), 11)
        x = np.tile(bits[:, None] * pn[None, :], (1, 1)).reshape(-1)[:n].astype(np.complex64)
    elif name == "OFDM":
        x = ofdm_256(rng, order=4)
    elif name == "OFDM_QPSK":
        x = ofdm_256(rng, order=4)
    elif name == "OFDM_16QAM":
        x = ofdm_256(rng, order=16)
    elif name == "ASK":
        amp = rng.choice(np.array([0.35, 0.75, 1.15, 1.55]), n)
        x = amp * np.exp(1j * 2 * np.pi * 55_000 * t)
    elif name == "OOK":
        amp = rng.integers(0, 2, n).astype(np.float32)
        x = amp * np.exp(1j * 2 * np.pi * 45_000 * t)
    elif name == "PAM4":
        amp = rng.choice(np.array([-3.0, -1.0, 1.0, 3.0]), n)
        x = amp * np.exp(1j * 2 * np.pi * 40_000 * t)
    elif name == "PAM8":
        amp = rng.choice(np.arange(-7, 8, 2, dtype=np.float32), n)
        x = amp * np.exp(1j * 2 * np.pi * 35_000 * t)
    elif name == "FHSS":
        x = np.zeros(n, dtype=np.complex64)
        hop_len = 32
        freqs = rng.choice(np.array([-120e3, -80e3, -40e3, 40e3, 80e3, 120e3]), int(np.ceil(n / hop_len)))
        phase0 = 0.0
        for k, start in enumerate(range(0, n, hop_len)):
            stop = min(start + hop_len, n)
            local_t = np.arange(stop - start) / (FS / UPSAMPLE)
            ph = phase0 + 2 * np.pi * freqs[k] * local_t
            x[start:stop] = np.exp(1j * ph)
            phase0 = float(ph[-1] + 2 * np.pi * freqs[k] / (FS / UPSAMPLE))
    elif name == "Radar_Pulse":
        x = np.zeros(n, dtype=np.complex64)
        for start in [12, 76, 140, 204]:
            length = min(24, n - start)
            tt = np.arange(length) / (FS / UPSAMPLE)
            pulse = chirp(tt, f0=-90e3, f1=90e3, t1=max(tt[-1], 1e-6), method="linear")
            x[start:start + length] = pulse.astype(np.complex64)
    elif name == "Burst_Signal":
        x = np.zeros(n, dtype=np.complex64)
        for _ in range(5):
            start = int(rng.integers(0, n - 20))
            length = int(rng.integers(8, 24))
            noise = rng.normal(size=length) + 1j * rng.normal(size=length)
            window = np.hanning(length)
            x[start:start + length] += (noise * window).astype(np.complex64)
    elif name == "LoRa_like":
        tt = np.arange(n) / (FS / UPSAMPLE)
        sweep_time = tt[-1] + 1 / (FS / UPSAMPLE)
        phase = 2 * np.pi * (-120e3 * tt + 0.5 * (240e3 / sweep_time) * tt ** 2)
        x = np.exp(1j * phase).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported modulation: {name}")

    return normalize_rms(x)



def make_tx_waveform(name: str, seed: int) -> np.ndarray:
    raw = generate_baseband(name, seed)
    active = np.repeat(raw, UPSAMPLE).astype(np.complex64)
    active = normalize_rms(active)
    scaled = active * TX_DIGITAL_RMS
    peak = float(np.max(np.abs(scaled)))
    if peak > 14000.0:
        scaled *= 14000.0 / peak
    return np.concatenate([
        np.zeros(GUARD, dtype=np.complex64),
        scaled.astype(np.complex64),
        np.zeros(GUARD, dtype=np.complex64),
    ]).astype(np.complex64)


def detect_and_reduce(capture: np.ndarray, noise: np.ndarray):
    capture = np.asarray(capture, dtype=np.complex64).reshape(-1)
    noise = np.asarray(noise, dtype=np.complex64).reshape(-1)
    capture_dc = capture - np.mean(capture)
    noise_dc = noise - np.mean(noise)
    noise_power = float(np.mean(np.abs(noise_dc) ** 2))
    capture_power = float(np.mean(np.abs(capture_dc) ** 2))
    power = np.abs(capture_dc) ** 2
    kernel = np.ones(ACTIVE_LEN, dtype=np.float64) / ACTIVE_LEN
    window_power = np.convolve(power, kernel, mode="valid")
    start = int(np.argmax(window_power))
    peak_window_power = float(window_power[start])
    ratio = peak_window_power / max(noise_power, 1e-20)
    ratio_db = float(10 * np.log10(max(ratio, 1e-20)))
    capture_ratio_db = float(10 * np.log10(
        max(capture_power, 1e-20) / max(noise_power, 1e-20)))
    found = bool(ratio >= DETECTION_MARGIN and start + ACTIVE_LEN <= len(capture_dc))
    reduced = np.zeros(RAW_LEN, dtype=np.complex64)
    if found:
        window = capture_dc[start:start + ACTIVE_LEN]
        reduced = window.reshape(RAW_LEN, UPSAMPLE).mean(axis=1).astype(np.complex64)
    return {
        "found": found, "start": start, "reduced": reduced,
        "noise_power": noise_power, "capture_power": capture_power,
        "capture_to_noise_db": capture_ratio_db,
        "peak_window_power": peak_window_power,
        "window_to_noise_db": ratio_db,
        "maximum_amplitude": float(np.max(np.abs(capture_dc))),
        "clipping_detected": bool(np.max(np.abs(capture)) >= 2000.0),
    }


def split_for_repetition(rep: int):
    if rep <= TRAIN_PER_COMBINATION:
        return "train", 1 + (rep - 1) // 50
    if rep <= TRAIN_PER_COMBINATION + VALIDATION_PER_COMBINATION:
        return "validation", 8 + (rep - TRAIN_PER_COMBINATION - 1) // 45
    return "test", 10 + (rep - TRAIN_PER_COMBINATION - VALIDATION_PER_COMBINATION - 1) // 45


def build_plan():
    """Build a deterministic, split-safe, strict TX round-robin plan.

    The active transmitter order is always Pluto_02, Pluto_04, Pluto_05,
    then repeats. Modulations and repetitions are shuffled independently
    inside each split and transmitter, while split membership remains fixed.
    """
    by_split_tx = {
        split_name: {tx_name: [] for tx_name, _ in TX_UNITS}
        for split_name in ("train", "validation", "test")
    }
    capture_index = 0
    for mod_name, class_name, class_label in MODULATIONS:
        for tx_name, tx_uri in TX_UNITS:
            for repetition in range(1, CAPTURES_PER_COMBINATION + 1):
                capture_index += 1
                split_name, session_id = split_for_repetition(repetition)
                by_split_tx[split_name][tx_name].append({
                    "capture_index": capture_index,
                    "modulation_name": mod_name,
                    "class_name": class_name,
                    "class_label": class_label,
                    "tx_unit": tx_name,
                    "tx_uri": tx_uri,
                    "repetition": repetition,
                    "split": split_name,
                    "session_id": session_id,
                    "random_seed": MASTER_SEED + capture_index * 1009,
                })

    rng = np.random.default_rng(MASTER_SEED)
    ordered = []
    for split_name in ("train", "validation", "test"):
        shuffled = {}
        for tx_name, _ in TX_UNITS:
            rows = by_split_tx[split_name][tx_name]
            perm = rng.permutation(len(rows))
            shuffled[tx_name] = [rows[i] for i in perm]
        rows_per_tx = len(shuffled[TX_UNITS[0][0]])
        if any(len(shuffled[name]) != rows_per_tx for name, _ in TX_UNITS):
            raise RuntimeError("TX plans are not balanced")
        for row_index in range(rows_per_tx):
            for tx_name, _ in TX_UNITS:
                ordered.append(shuffled[tx_name][row_index])

    for order, row in enumerate(ordered, start=1):
        row["capture_order"] = order
    assert len(ordered) == TARGET_CAPTURES
    for i, row in enumerate(ordered):
        expected_tx = TX_UNITS[i % len(TX_UNITS)][0]
        if row["tx_unit"] != expected_tx:
            raise RuntimeError("Round-robin TX ordering validation failed")
    return ordered


def paths(output_dir: Path):
    return {
        "iq": output_dir / "real_iq_31800.npy",
        "ok": output_dir / "captured_ok_31800.npy",
        "records": output_dir / "ground_truth_31800.csv",
        "plan_json": output_dir / "capture_plan_31800.json",
        "manifest": output_dir / "manifest_31800.json",
        "final_mat": output_dir / "real_sdr_dataset_31800.mat",
    }


def write_records(path: Path, records_by_order: dict):
    if not records_by_order:
        return
    rows = [records_by_order[k] for k in sorted(records_by_order)]
    temp = path.with_suffix(".tmp.csv")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def save_manifest(output_dir: Path, ok: np.ndarray, records_by_order: dict, complete=False):
    p = paths(output_dir)
    manifest = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "target_captures": TARGET_CAPTURES,
        "successful_captures": int(np.sum(ok)),
        "failed_or_pending": int(TARGET_CAPTURES - np.sum(ok)),
        "recorded_rows": len(records_by_order),
        "complete": bool(complete),
        "sample_shape": [TARGET_CAPTURES, RAW_LEN, 2],
        "center_frequency_hz": FC,
        "sample_rate_hz": FS,
        "tx_gain_db": TX_GAIN_DB,
        "rx_gain_db": RX_GAIN_DB,
        "tx_digital_rms": TX_DIGITAL_RMS,
        "detection_margin_linear": DETECTION_MARGIN,
        "train_count": 21000,
        "validation_count": 5400,
        "test_count": 5400,
    }
    temp = p["manifest"].with_suffix(".tmp.json")
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temp, p["manifest"])


def load_or_create(output_dir: Path):
    p = paths(output_dir)
    if p["iq"].exists():
        iq = np.lib.format.open_memmap(p["iq"], mode="r+")
        ok = np.load(p["ok"], mmap_mode="r+")
    else:
        iq = np.lib.format.open_memmap(
            p["iq"], mode="w+", dtype=np.float32,
            shape=(TARGET_CAPTURES, RAW_LEN, 2))
        iq[:] = 0
        ok = np.lib.format.open_memmap(
            p["ok"], mode="w+", dtype=np.uint8,
            shape=(TARGET_CAPTURES,))
        ok[:] = 0
        iq.flush(); ok.flush()
    records = {}
    if p["records"].exists():
        with p["records"].open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                records[int(row["capture_order"])] = row
    return iq, ok, records


def configure_radio(radio, is_tx: bool):
    radio.sample_rate = FS
    if is_tx:
        radio.tx_lo = FC
        radio.tx_rf_bandwidth = FS
        radio.tx_hardwaregain_chan0 = TX_GAIN_DB
        radio.tx_cyclic_buffer = True
    else:
        radio.rx_lo = FC
        radio.rx_rf_bandwidth = FS
        radio.rx_buffer_size = CAPTURE_LEN
        radio.gain_control_mode_chan0 = "manual"
        radio.rx_hardwaregain_chan0 = RX_GAIN_DB


def finalize_mat(output_dir: Path, plan, iq, ok):
    p = paths(output_dir)
    successful = np.asarray(ok, dtype=bool)
    if not np.all(successful):
        return
    real_iq = np.asarray(iq, dtype=np.float32)
    savemat(p["final_mat"], {
        "realIQ": real_iq,
        "capturedOk": np.asarray(ok, dtype=np.uint8),
        "modulationName": np.array([r["modulation_name"] for r in plan], dtype=object),
        "className": np.array([r["class_name"] for r in plan], dtype=object),
        "classLabel": np.array([r["class_label"] for r in plan], dtype=np.int16),
        "txUnit": np.array([r["tx_unit"] for r in plan], dtype=object),
        "split": np.array([r["split"] for r in plan], dtype=object),
        "sessionID": np.array([r["session_id"] for r in plan], dtype=np.int16),
        "captureOrder": np.array([r["capture_order"] for r in plan], dtype=np.int32),
        "centerFrequency": np.array([[FC]], dtype=np.float64),
        "sampleRate": np.array([[FS]], dtype=np.float64),
    }, do_compression=True)


def format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m {secs:02d}s"
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"


def print_eta(ok: np.ndarray, initial_success_count: int, run_start_monotonic: float):
    successful_now = int(np.sum(ok))
    gained = successful_now - initial_success_count
    elapsed = max(time.monotonic() - run_start_monotonic, 1e-9)
    remaining = TARGET_CAPTURES - successful_now
    if gained > 0:
        seconds_per_success = elapsed / gained
        eta_seconds = remaining * seconds_per_success
        rate_per_hour = 3600.0 / seconds_per_success
        print(
            f"PROGRESS | successful={successful_now}/{TARGET_CAPTURES} | "
            f"remaining={remaining} | elapsed={format_duration(elapsed)} | "
            f"rate={rate_per_hour:.1f} captures/hour | "
            f"estimated remaining={format_duration(eta_seconds)}"
        )
    else:
        print(
            f"PROGRESS | successful={successful_now}/{TARGET_CAPTURES} | "
            f"remaining={remaining} | elapsed={format_duration(elapsed)} | "
            "estimated remaining=calculating"
        )


def request_stop(signum=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\nSTOP REQUEST RECEIVED. Finishing the current capture and saving checkpoint...")


def stop_file_requested(output_dir: Path) -> bool:
    return (output_dir / STOP_FILE_NAME).exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libiio-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-captures", type=int, default=0,
                        help="0 means continue until all rows are successful")
    args = parser.parse_args()

    # Make libiio visible to both Windows DLL loading and ctypes.find_library.
    # Keep the directory handle alive for the whole process.
    libiio_folder = str(Path(args.libiio_folder).resolve())
    libiio_dll = str(Path(libiio_folder) / "libiio.dll")
    if not Path(libiio_dll).is_file():
        raise RuntimeError(f"libiio.dll was not found: {libiio_dll}")
    os.environ["PATH"] = libiio_folder + os.pathsep + os.environ.get("PATH", "")
    dll_directory_handle = os.add_dll_directory(libiio_folder)
    import ctypes
    libiio_preload_handle = ctypes.WinDLL(libiio_dll)
    import adi

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_file = output_dir / STOP_FILE_NAME
    if stop_file.exists():
        stop_file.unlink()
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    plan = build_plan()
    p = paths(output_dir)
    if not p["plan_json"].exists():
        p["plan_json"].write_text(json.dumps(plan, indent=2), encoding="utf-8")
    iq, ok, records = load_or_create(output_dir)
    initial_success_count = int(np.sum(ok))
    run_start_monotonic = time.monotonic()
    print(f"RESUME: {initial_success_count}/{TARGET_CAPTURES} successful")
    print("Connecting to receiver...")
    print("TX order: Pluto_02 -> Pluto_04 -> Pluto_05 -> repeat.")
    print(f"Each physical capture duration: {SETTLE_SECONDS:.1f} seconds.")
    print("Each transmitter is opened fresh for every retry.")
    print("Safe stop: press Ctrl+C once; current capture finishes, then checkpoint is saved.")
    print(f"Alternative safe stop file: {stop_file}")
    rx = adi.Pluto(uri=RX_URI)
    configure_radio(rx, is_tx=False)
    new_successes = 0
    processed_since_checkpoint = 0
    try:
        print("RX warm-up...")
        for _ in range(3):
            rx.rx()
        for row in plan:
            if STOP_REQUESTED or stop_file_requested(output_dir):
                print("SAFE STOP: no new capture will be started.")
                break
            order = int(row["capture_order"])
            idx = order - 1
            if ok[idx]:
                continue
            if args.max_new_captures > 0 and new_successes >= args.max_new_captures:
                break
            print("\n" + "-" * 76)
            print(f"Capture {order:05d}/{TARGET_CAPTURES} | {row['split']} | "
                  f"{row['modulation_name']} | {row['class_name']} | TX={row['tx_unit']}")
            result = None
            message = ""
            attempts = 0
            for attempt in range(1, MAX_RETRIES + 1):
                attempts = attempt
                tx = None
                try:
                    tx = adi.Pluto(uri=row["tx_uri"])
                    configure_radio(tx, is_tx=True)
                    noise = np.asarray(rx.rx(), dtype=np.complex64)
                    waveform = make_tx_waveform(
                        row["modulation_name"], int(row["random_seed"]) + attempt - 1)
                    tx.tx(waveform)
                    time.sleep(SETTLE_SECONDS)
                    for _ in range(RX_FLUSH_BUFFERS_AFTER_TX):
                        rx.rx()
                    capture = np.asarray(rx.rx(), dtype=np.complex64)
                    result = detect_and_reduce(capture, noise)
                    if result["found"]:
                        message = "OK"
                        break
                    message = "Burst not detected"
                    print(f"Attempt {attempt}: not detected | "
                          f"window/noise={result['window_to_noise_db']:.2f} dB")
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    print(f"Attempt {attempt} failed: {message}")
                finally:
                    if tx is not None:
                        try: tx.tx_destroy_buffer()
                        except Exception: pass
                        del tx
                    time.sleep(0.20)
            success = bool(result is not None and result["found"])
            if success:
                reduced = result["reduced"]
                iq[idx, :, 0] = reduced.real.astype(np.float32)
                iq[idx, :, 1] = reduced.imag.astype(np.float32)
                ok[idx] = 1
                new_successes += 1
                print(f"CAPTURED | capture/noise={result['capture_to_noise_db']:.2f} dB | "
                      f"window/noise={result['window_to_noise_db']:.2f} dB | start={result['start']}")
            else:
                if result is None:
                    result = {"noise_power": np.nan, "capture_power": np.nan,
                              "capture_to_noise_db": np.nan, "peak_window_power": np.nan,
                              "window_to_noise_db": np.nan, "maximum_amplitude": np.nan,
                              "clipping_detected": False, "start": -1}
                print(f"FAILED after {attempts} attempts: {message}")
            record = {
                **row,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "rx_unit": RX_NAME, "rx_uri": RX_URI,
                "attempts": attempts, "captured_ok": int(success), "message": message,
                "center_frequency_hz": FC, "sample_rate_hz": FS,
                "tx_gain_db": TX_GAIN_DB, "rx_gain_db": RX_GAIN_DB,
                "tx_digital_rms": TX_DIGITAL_RMS,
                "noise_floor_power": result["noise_power"],
                "capture_power": result["capture_power"],
                "capture_to_noise_db": result["capture_to_noise_db"],
                "peak_window_power": result["peak_window_power"],
                "window_to_noise_db": result["window_to_noise_db"],
                "maximum_amplitude": result["maximum_amplitude"],
                "clipping_detected": int(result["clipping_detected"]),
                "burst_start_sample": result["start"],
            }
            records[order] = record
            processed_since_checkpoint += 1
            print_eta(ok, initial_success_count, run_start_monotonic)
            if processed_since_checkpoint >= CHECKPOINT_INTERVAL or not success:
                iq.flush(); ok.flush()
                write_records(p["records"], records)
                save_manifest(output_dir, ok, records)
                processed_since_checkpoint = 0
                print(f"CHECKPOINT | successful={int(np.sum(ok))}/{TARGET_CAPTURES} | "
                      f"recorded={len(records)}/{TARGET_CAPTURES}")
            if STOP_REQUESTED or stop_file_requested(output_dir):
                print("SAFE STOP: current capture saved; stopping before the next capture.")
                break
    finally:
        print("\nClosing receiver...")
        del rx
        iq.flush(); ok.flush()
        write_records(p["records"], records)
        complete = bool(np.all(np.asarray(ok, dtype=bool)))
        save_manifest(output_dir, ok, records, complete=complete)
        if stop_file.exists():
            try:
                stop_file.unlink()
            except OSError:
                pass

    success_count = int(np.sum(ok))
    print("\n" + "=" * 76)
    print("REAL HARDWARE DATASET COLLECTION STATUS")
    print("=" * 76)
    print(f"Successful captures: {success_count}/{TARGET_CAPTURES}")
    print(f"Remaining captures: {TARGET_CAPTURES - success_count}")
    print(f"Output folder: {output_dir}")
    print(f"Current run elapsed: {format_duration(time.monotonic() - run_start_monotonic)}")
    if STOP_REQUESTED:
        print("Stopped safely by user request. Checkpoint and ground truth were saved.")
    if success_count == TARGET_CAPTURES:
        print("Finalizing MATLAB dataset file...")
        finalize_mat(output_dir, plan, iq, ok)
        save_manifest(output_dir, ok, records, complete=True)
        print("COLLECTION COMPLETE")
    else:
        print("Run the collector again to resume failed or pending rows.")
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
