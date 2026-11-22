"""
Synthetic baseband IQ waveform generators.

The occupied-signal generators produce 256 complex samples with unit mean
sample power, mean(abs(x)**2) = 1, before H0/H1 observation scaling. Selected
waveform parameters vary between draws.

In build_observation(), H0 contains unit-variance complex Gaussian noise.
Under H1, the waveform is scaled to the requested SNR and added to the noise.
The resulting observation retains its measured sample power.
"""

import numpy as np


N_SAMPLES = 256
RNG_SEED_ENTROPY = 12345


def _rrc_pulse(beta, span, sps):
    """Root-raised-cosine pulse shaping filter."""
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(4 * beta * ti) - 1.0) < 1e-8:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


def _shape_and_resample(symbols, sps, rng, beta=None):
    if beta is None:
        beta = rng.uniform(0.25, 0.45)
    pulse = _rrc_pulse(beta, span=4, sps=sps)
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    shaped = np.convolve(up, pulse, mode="same")
    return shaped


def _fit_length(x, n=N_SAMPLES):
    if len(x) >= n:
        start = (len(x) - n) // 2
        return x[start:start + n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def _norm_energy(x):
    e = np.mean(np.abs(x) ** 2)
    if e < 1e-12:
        return x
    return x / np.sqrt(e)


def _phase_jitter(x, rng):
    phi0 = rng.uniform(0, 2 * np.pi)
    # Apply one constant phase offset to the entire window.
    return x * np.exp(1j * phi0)


def gen_psk(order, rng, sps_range=(3, 5)):
    sps = rng.integers(*sps_range)
    nsym = int(np.ceil(N_SAMPLES / sps)) + 8
    bits = rng.integers(0, order, size=nsym)
    syms = np.exp(1j * 2 * np.pi * bits / order + 1j * rng.uniform(0, 2 * np.pi))
    x = _shape_and_resample(syms, sps, rng)
    return _norm_energy(_fit_length(x))


def gen_qam(order, rng, sps_range=(3, 5)):
    sps = rng.integers(*sps_range)
    side = int(np.sqrt(order))
    nsym = int(np.ceil(N_SAMPLES / sps)) + 8
    levels = np.arange(side) * 2 - (side - 1)
    ib = levels[rng.integers(0, side, size=nsym)]
    qb = levels[rng.integers(0, side, size=nsym)]
    syms = (ib + 1j * qb).astype(complex)
    syms /= np.sqrt(np.mean(np.abs(syms) ** 2))
    x = _shape_and_resample(syms, sps, rng)
    return _norm_energy(_fit_length(x))


def gen_ofdm(rng, nfft=64, cp=16, qam_order=4, nsym=6):
    side = int(np.sqrt(qam_order)) if qam_order > 2 else 2
    x_all = []
    for _ in range(nsym):
        if qam_order == 2:
            data = np.exp(1j * np.pi * rng.integers(0, 2, size=nfft))
        else:
            levels = np.arange(side) * 2 - (side - 1)
            ib = levels[rng.integers(0, side, size=nfft)]
            qb = levels[rng.integers(0, side, size=nfft)]
            data = (ib + 1j * qb).astype(complex)
        data /= np.sqrt(np.mean(np.abs(data) ** 2) + 1e-12)
        td = np.fft.ifft(data) * np.sqrt(nfft)
        sym = np.concatenate([td[-cp:], td])
        x_all.append(sym)
    x = np.concatenate(x_all)
    return _norm_energy(_fit_length(x))


def gen_gfsk(rng, bt=0.35):
    sps = rng.integers(4, 7)
    nsym = int(np.ceil(N_SAMPLES / sps)) + 8
    bits = rng.integers(0, 2, size=nsym) * 2 - 1
    up = np.repeat(bits, sps).astype(float)
    span = 3
    t = np.arange(-span * sps, span * sps + 1) / sps
    g = np.exp(-(2 * np.pi ** 2 * bt ** 2) * t ** 2)
    g /= np.sum(g)
    freq = np.convolve(up, g, mode="same")
    h_mod = rng.uniform(0.4, 0.7)
    phase = np.cumsum(freq) * (np.pi * h_mod / sps)
    x = np.exp(1j * phase)
    return _norm_energy(_fit_length(x))


def gen_cpfsk(rng):
    return gen_gfsk(rng, bt=1.0)  # no Gaussian smoothing => plain CPFSK-like


def gen_msk(rng):
    sps = rng.integers(4, 7)
    nsym = int(np.ceil(N_SAMPLES / sps)) + 8
    bits = rng.integers(0, 2, size=nsym) * 2 - 1
    up = np.repeat(bits, sps).astype(float)
    phase = np.cumsum(up) * (np.pi / (2 * sps))
    x = np.exp(1j * phase)
    return _norm_energy(_fit_length(x))


def gen_ook(rng):
    sps = rng.integers(3, 6)
    nsym = int(np.ceil(N_SAMPLES / sps)) + 8
    bits = rng.integers(0, 2, size=nsym).astype(float)
    syms = bits.astype(complex)
    x = _shape_and_resample(syms, sps, rng, beta=rng.uniform(0.3, 0.5))
    return _norm_energy(_fit_length(x))


def gen_ask(rng, nlevels=4):
    sps = rng.integers(3, 6)
    nsym = int(np.ceil(N_SAMPLES / sps)) + 8
    levels = np.linspace(0.25, 1.0, nlevels)
    syms = levels[rng.integers(0, nlevels, size=nsym)].astype(complex)
    x = _shape_and_resample(syms, sps, rng)
    return _norm_energy(_fit_length(x))


def gen_pam(order, rng):
    sps = rng.integers(3, 6)
    nsym = int(np.ceil(N_SAMPLES / sps)) + 8
    levels = np.linspace(-1, 1, order)
    syms = levels[rng.integers(0, order, size=nsym)].astype(complex)
    x = _shape_and_resample(syms, sps, rng)
    return _norm_energy(_fit_length(x))


def gen_dsss(rng, chip_ratio=8):
    nsym = int(np.ceil(N_SAMPLES / chip_ratio)) + 4
    bits = rng.integers(0, 2, size=nsym) * 2 - 1
    pn = rng.integers(0, 2, size=chip_ratio) * 2 - 1
    chips = np.repeat(bits, chip_ratio) * np.tile(pn, nsym)
    x = chips.astype(complex)
    x = _phase_jitter(x, rng)
    return _norm_energy(_fit_length(x))


def gen_fhss_symbol(rng):
    # short-burst tone hopping within-symbol (abstracted as filtered noise-like burst)
    return gen_ook(rng)


def gen_lora(rng, bw_frac=0.5, sf=7):
    n = N_SAMPLES
    t = np.arange(n)
    k = rng.uniform(0.3, 0.7) * bw_frac
    f0 = rng.uniform(-0.4, 0.4)
    phase = 2 * np.pi * (f0 * t + 0.5 * k * (t ** 2) / n)
    x = np.exp(1j * phase)
    return _norm_energy(x)


def gen_radar_pulse(rng, duty=0.3):
    n = N_SAMPLES
    pulse_len = max(4, int(n * duty * rng.uniform(0.7, 1.3)))
    period = rng.integers(pulse_len + 4, pulse_len + 40)
    x = np.zeros(n, dtype=complex)
    t = 0
    chirp_rate = rng.uniform(-1, 1)
    while t < n:
        seg = min(pulse_len, n - t)
        tt = np.arange(seg)
        x[t:t + seg] = np.exp(
            1j * (2 * np.pi * chirp_rate * tt ** 2 / (2 * pulse_len))
        )
        t += period
    return _norm_energy(x)


def gen_burst(rng, duty=None):
    n = N_SAMPLES
    if duty is None:
        duty = rng.uniform(0.2, 0.5)
    burst_len = max(8, int(n * duty))
    start = rng.integers(0, n - burst_len + 1)
    core = gen_qam(16, rng)
    x = np.zeros(n, dtype=complex)
    x[start:start + burst_len] = core[:burst_len]
    return _norm_energy(x if np.any(x) else core)


SYSTEM_GENERATORS = {
    "BPSK": lambda rng: gen_psk(2, rng),
    "QPSK": lambda rng: gen_psk(4, rng),
    "8PSK": lambda rng: gen_psk(8, rng),
    "16QAM": lambda rng: gen_qam(16, rng),
    "64QAM": lambda rng: gen_qam(64, rng),
    "OFDM": lambda rng: gen_ofdm(rng, qam_order=2),
    "OFDM_QPSK": lambda rng: gen_ofdm(rng, qam_order=4),
    "OFDM_16QAM": lambda rng: gen_ofdm(rng, qam_order=16),
    "GFSK": lambda rng: gen_gfsk(rng),
    "CPFSK": lambda rng: gen_cpfsk(rng),
    "MSK": lambda rng: gen_msk(rng),
    "OOK": lambda rng: gen_ook(rng),
    "ASK": lambda rng: gen_ask(rng),
    "PAM4": lambda rng: gen_pam(4, rng),
    "PAM8": lambda rng: gen_pam(8, rng),
    "DSSS": lambda rng: gen_dsss(rng),
    "FHSS": lambda rng: gen_fhss_symbol(rng),
    "LoRa_like": lambda rng: gen_lora(rng),
    "Radar_Pulse": lambda rng: gen_radar_pulse(rng),
    "Burst_Signal": lambda rng: gen_burst(rng),
}

SYSTEM_NAMES = list(SYSTEM_GENERATORS.keys())  # Index 0..19 maps to system_id 1..20.


def gen_idle_noise(rng):
    """Pure noise reference shape (unit variance complex Gaussian) -- H0."""
    x = (rng.normal(size=N_SAMPLES) + 1j * rng.normal(size=N_SAMPLES)) / np.sqrt(2)
    return x


def build_observation(rng, system_name, snr_db, is_idle=False):
    """
    Implements the H0/H1 model of eq. 2:
        y_i = w_i                      (H0, idle)
        y_i = sqrt(Pt) s_i + w_i        (H1, occupied)
    Noise power sigma_w^2 is fixed at 1.0. Pt is set from snr_db relative
    to that fixed noise floor. The output retains its sample power.
    """
    sigma_w2 = 1.0
    w = np.sqrt(sigma_w2 / 2) * (
        rng.normal(size=N_SAMPLES) + 1j * rng.normal(size=N_SAMPLES)
    )
    if is_idle:
        y = w
    else:
        s = SYSTEM_GENERATORS[system_name](rng)
        pt = sigma_w2 * 10 ** (snr_db / 10.0)
        y = np.sqrt(pt) * s + w
    iq = np.stack([y.real, y.imag], axis=-1).astype(np.float32)
    return iq