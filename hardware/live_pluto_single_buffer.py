import os
import sys

libiio_folder = sys.argv[1]
uri = sys.argv[2]
center_frequency = int(sys.argv[3])
sample_rate = int(sys.argv[4])
buffer_size = int(sys.argv[5])
output_file = sys.argv[6]

os.add_dll_directory(libiio_folder)

import adi
import numpy as np

sdr = adi.Pluto(uri=uri)
sdr.rx_lo = center_frequency
sdr.sample_rate = sample_rate
sdr.rx_rf_bandwidth = sample_rate
sdr.rx_buffer_size = buffer_size
sdr.gain_control_mode_chan0 = "slow_attack"

for _ in range(3):
    sdr.rx()

iq = np.asarray(sdr.rx(), dtype=np.complex64)
iq.tofile(output_file)

print("REAL_RX_SAMPLES:", len(iq))
