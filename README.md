# Signal-profile classification for frequency-hopping spectrum monitoring

Code for the paper *Signal-Profile Classification for Frequency-Hopping Spectrum Monitoring: A Cross-Device Software-Defined Radio Study* (submitted to Physical Communication).

A two-stage design detects occupied channels with an energy detector and assigns each occupied 256-sample IQ observation to a friendly or an adversarial signal profile with a CNN–LSTM classifier. The code covers the synthetic frequency-hopping experiments, the capture of 31,800 PlutoSDR observations from three transmitters, leave-one-device-out training, live conducted tests, and the analyses reported in the paper.

## Data

The datasets are too large for GitHub and are archived on Zenodo:

**https://doi.org/[DOI]**

Download the files into one folder. That folder is the *data folder* used below.

| File on Zenodo | Contents |
|---|---|
| `dataset_v5.mat` | Synthetic Dataset v5 (31,680 observations, labels, SNR, splits, FH annotations) |
| `real_sdr_dataset_31800.mat` | 31,800 PlutoSDR observations with modulation, class, transmitter, split and session labels |
| `real_sdr_dataset_31800_clean_split.mat` | Leakage-controlled split used for training |
| `ground_truth_31800.csv` | Per-capture acquisition metadata |

## Contents

| Folder | Script | Paper section |
|---|---|---|
| `synthetic/` | `prepare_occupied_sequences_v5.m` | 5.1, occupied subset for Stage 2 |
| | `inspect_fh_dataset_v5.m` | 5.1, dataset inspection |
| | `energy_threshold_sensing_v5.m` | 4.1 and 6.3, Stage 1 threshold |
| | `train_cnn_lstm_occ_v5.m` | 4.5 and 6.2, synthetic CNN–LSTM |
| | `regenerate_snr_binned_cnn_lstm_v5.m` | 6.3, SNR-binned results |
| `hardware/` | `capture_real_dataset_31800_round_robin.py` (and `.m` wrapper) | 5.3, dataset capture |
| | `build_clean_split_31800_v2.m` | 5.2, leakage-controlled split |
| | `analyze_cross_split_leakage_31800.m`, `findCrossSplitMaximums.m`, `inspect_temporal_similarity_31800.m` | 5.4, near-duplicate screen |
| | `train_real_cnn_lstm_31800_v1.m` | 5.6 and 6.4, leave-one-device-out training |
| | `run_live_end_to_end_hardware_v1.m`, `live_pluto_single_buffer.py` | 5.5 and 6.5, live conducted tests |
| `analysis/` | `pipeline_end_to_end.py` | 6.3, both stages on the same test observations |
| | `fig_energy_detector.py` | 6.3, energy-detector figure |
| | `cfo_offsets.py` | 6.6, carrier-frequency offset per transmitter |
| | `mat73.py` | Reader for MATLAB 7.3 files (used by the scripts above) |
| `results/` | CSV and MAT files | Outputs behind the reported numbers |

## Requirements

- MATLAB with the Deep Learning Toolbox for the `.m` scripts.
- Python 3 with the packages in `requirements.txt`. `pyadi-iio` is needed only for the capture scripts, which also require PlutoSDR hardware.

## Running

MATLAB scripts read and write files in the current folder, so change to the data folder first and add the repository folders to the path:

```matlab
cd('path/to/data_folder')
addpath('path/to/repo/synthetic', 'path/to/repo/hardware')
train_cnn_lstm_occ_v5
```

The Python analyses take the data folder as an argument. Copy the files from `results/` into the data folder first.

```bash
python analysis/pipeline_end_to_end.py path/to/data_folder
python analysis/fig_energy_detector.py path/to/data_folder
python analysis/cfo_offsets.py path/to/data_folder/real_sdr_dataset_31800.mat
```

`pipeline_end_to_end.py` first recomputes the Stage 1 energies and stops unless they reproduce the saved Stage 1 confusion matrix exactly.

## License

Code: MIT License (see `LICENSE`). Data on Zenodo: CC BY 4.0.
