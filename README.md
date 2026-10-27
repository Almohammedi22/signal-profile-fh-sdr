# Signal-profile classification for spectrum monitoring

Code and data for the paper *Friendly and Adversarial Signal-Profile Classification for Spectrum Monitoring: Cross-Transmitter Validation with Software-Defined Radios*.

A two-stage design detects occupied channels with an energy detector and assigns each occupied 256-sample IQ window to a friendly or an adversarial signal profile with a deep-learning classifier. The repository covers the synthetic experiments (Dataset v5), the capture of 31,800 PlutoSDR windows from three transmitters, leave-one-device-out training, the live conducted tests, and every analysis reported in the paper.

## Data

The data files are attached to release **v2.0** of this repository (Releases, on the right of the repository page). Download them and unzip them into one folder, the *data folder*.

| Release file | Contents |
|---|---|
| `data_synthetic_v5.zip` | `dataset_v5.mat`: synthetic Dataset v5 (31,680 windows, labels, SNR, episode partition, FH annotations) and `energy_gate_v5.mat` (Stage 1 threshold) |
| `data_hardware_31800.zip` | `real_sdr_dataset_31800.mat` (31,800 PlutoSDR windows with waveform, class, transmitter, session and capture order), `clean_split_full_metadata.csv` (leakage-controlled partition), `leakage_components.csv`, `ground_truth_31800.csv` (per-capture acquisition metadata) |
| `runs_v2.zip` | One folder per training run reported in the paper: configuration and training history (`info.json`), per-window class probabilities (`scores.npz`), trained parameters (`weights.npz`), training log, and the frequency-shift results of the hardware runs (`cfo_shift.json`) |

## Contents

| Folder | Contents |
|---|---|
| `replication/` | NumPy implementation of the six networks and the complete pipeline used in the paper (see below) |
| `synthetic/` | Original MATLAB scripts for Dataset v5 (occupied subset, energy threshold, CNN-LSTM training, SNR-binned results) |
| `hardware/` | Capture of the 31,800 windows, leakage-controlled split, MATLAB leave-one-device-out training, live conducted tests |
| `analysis/` | Supporting analyses of the MATLAB experiments (end-to-end pipeline, energy-detector figure, carrier-frequency offsets) |
| `results/` | Summary tables reported in the paper (CSV and JSON) |

### `replication/`

| Script | Purpose | Paper |
|---|---|---|
| `config.py` | Folder locations (data, cache, runs, results, figures); override with environment variables | |
| `prepare_cache.py` | Reads the MAT and CSV files once and writes float32 caches | 5.1-5.3 |
| `npdl.py`, `models.py` | Layers, losses, optimizer, the six architectures, training loop | 4.3, 4.5, Table 2 |
| `gradcheck.py` | Finite-difference check of the gradients of all six networks | 4.5 |
| `complexity.py`, `latency.py` | Parameters, multiply-accumulate operations, CPU time per window | Table 2, 6.2 |
| `run_experiment.py` | One training run (synthetic or hardware fold, real-only or combined, optional offset augmentation, 2 or 20 classes) | 5.4, 6.2-6.7 |
| `run_queue.py`, `queue.txt` | Runs all 110 training jobs of the paper; several copies can share the queue | |
| `cfo_shift_eval.py` | Test-time frequency-shift intervention on trained hardware models | 6.7 |
| `stage1_analysis.py` | Energy detector: theory, noise-power uncertainty, noise-referenced detector | 6.1 |
| `syn_measure.py`, `hw_diagnostics.py` | Properties measured on the stored windows; tests of alternative explanations | 5.1, 5.3, 6.7 |
| `features_baseline.py` | Hand-crafted-feature baselines | 6.2 |
| `analyze_results.py` | All metrics reported in the paper from the stored per-window predictions | 6.2-6.7 |
| `figs_results.py` | Result figures | Figs. 5-7 |

## Requirements

Python 3.10 or later with the packages in `requirements.txt`. The replication code uses only NumPy, SciPy, pandas, scikit-learn (baselines) and Matplotlib (figures). MATLAB with the Deep Learning Toolbox is needed only for the original `.m` scripts, and `pyadi-iio` with PlutoSDR hardware only for the capture scripts.

## Reproducing the reported results

```bash
cd replication
export SPR_DATA=/path/to/data_folder        # folder with the unzipped release files
python prepare_cache.py                     # once, about a minute
python gradcheck.py                         # optional
python run_queue.py 600 &                   # trains the 110 jobs of queue.txt; start one copy per CPU core
python cfo_shift_eval.py                    # after the hardware runs have finished
python stage1_analysis.py
python syn_measure.py
python hw_diagnostics.py
python features_baseline.py
python analyze_results.py                   # writes the summary tables to results/paper
python figs_results.py
```

Set `OPENBLAS_NUM_THREADS=1` when several queue workers run in parallel. To check the reported numbers without retraining, unzip `runs_v2.zip` into the runs folder (default `replication/work/runs`) and run only the analysis steps.

## License

Code: MIT License (see `LICENSE`). Data: CC BY 4.0.
