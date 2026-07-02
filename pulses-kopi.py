import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
import random
from joblib import Parallel, delayed
import time



# Echoes weaker than this many dB relative to the noise are inaudible in the dB
# spectrogram, so the reverberation loop stops once it drops below this floor.
SNR_FLOOR_DB = -40


def TL(f, r):

    """
    Calculates the transmission loss in dB as a function of frequency and range, accounting for spherical spreading
    (20*log10(r), referenced to 1 m) and frequency-dependent absorption in seawater.

    ----------

    Parameters:
        f (float or ndarray) - frequency in Hz.
        r (float or ndarray) - range from the source in meters.

    Returns:
        loss (float or ndarray) - transmission loss in dB.
    """

    f = f/1000  # Hz -> kHz

    alpha = (3.3 * 10**(-3) + 0.11 * f**2 / (1 + f**2) + 44 * f**2 / (4100 + f**2) + 3 * 10**(-4) * f**2) * 1/1000  # (Thorp, 1967) [dB/m]
    spherical = 20*np.log10(r)  # spherical spreading

    loss = spherical + alpha*r

    return loss


def rms(data):

    """
    Computes the root-mean-square (RMS) value of an array.

    ----------

    Parameters:
        data (ndarray) - array containing the values to compute the RMS of.

    Returns:
        (float) - the root-mean-square value of the input array.
    """

    return np.sqrt(np.mean(np.square(data)))


def intervals_overlap(start1, end1, start2, end2):

    """
    Checks whether two intervals overlap.

    ----------

    Parameters:
        start1 (float) - start of the first interval.
        end1 (float) - end of the first interval.
        start2 (float) - start of the second interval.
        end2 (float) - end of the second interval.

    Returns:
        (bool) - True if the intervals overlap, False otherwise.
    """

    return start1 < end2 and start2 < end1


def db(z,eps=1e-6):

    """
    Converts a magnitude input to decibels using 20*log10(|z|) with a reference of 1.0.
    A small offset (1e-6) is added to the magnitude to avoid taking the logarithm of zero.

    ----------

    Parameters:
        z (float, complex, or ndarray) - input value(s) to convert to dB. Can be real or complex.

    Returns:
        (float or ndarray) - the input expressed in decibels.
    """

    return 20 * np.log10(np.abs(z) + eps)


def EOS():

    """
    Creates an end-of-sequence (EOS) token used to mark the end of a target sequence or an empty sequence.
    The token is a flat 8-element array structured as [cw_flag, lfm_flag, hfm_flag, eos_flag, t_start, t_stop, f1, f2],
    with only the EOS flag set to 1 and all other entries equal to 0. 
    

    ----------

    Parameters:
        None

    Returns:
        eos (ndarray) - 1-D float64 array of length 8 representing the EOS token.
    """

    eos = np.zeros(8)
    eos[3] = 1  # EOS flag

    return eos


def pulse(noise_rms, snr, type, t_start, t_stop, t_window, freq0, freq1, fs, dist, reflections=True):

    """
    Generates a synthetic acoustic pulse embedded in a noise window, with an optional series of
    reverberation echoes. The pulse is a frequency-modulated chirp shaped by a Tukey window,
    with amplitude set by the desired SNR relative to the provided noise RMS. When reflections are
    enabled, additional attenuated echoes are added at increasing ranges, with arrival times computed
    from the round-trip travel distance and a fixed sound speed of 1500 m/s, and amplitudes reduced
    by the transmission loss (TL) over the corresponding distance.

    ----------

    Parameters:
        noise_rms (float) - RMS amplitude of the background noise used as the SNR reference.
        snr (float) - signal-to-noise ratio of the pulse in dB.
        type (str) - chirp method passed to scipy.signal.chirp ('linear' or 'hyperbolic').
        t_start (float) - start time of the pulse within the window, in seconds.
        t_stop (float) - end time of the pulse within the window, in seconds.
        t_window (float) - total duration of the output window, in seconds.
        freq0 (float) - starting frequency of the chirp in Hz.
        freq1 (float) - ending frequency of the chirp in Hz.
        fs (int) - sampling frequency in Hz.
        dist (float) - distance from the source to the receiver in meters.
        reflections (bool) - default set to True. If True, reverberation echoes are added; if False,
                             only the direct pulse is generated.

    Returns:
        y (ndarray) - 1-D array of length int(fs * t_window) containing the synthetic pulse signal.
    """

    total_samples = int(fs * t_window)
    pulse_duration = t_stop - t_start
    pulse_samples = int(fs * pulse_duration)

    t_pulse = np.linspace(0, pulse_duration, pulse_samples)

    y = np.zeros(total_samples)

    c = 1500
    r = dist
    amp = noise_rms * 10**(snr/20)

    noise = np.random.normal(size=pulse_samples)
    phase = np.random.uniform(0, 360)

    if reflections == False:

        window = sps.windows.tukey(pulse_samples, alpha=0.2)
        x = sps.chirp(t_pulse, freq0, pulse_duration, freq1, method=type, phi=phase)
        x = (noise + x) * window
        start_idx = int(round(t_start * fs))
        end_idx = start_idx + pulse_samples

        if end_idx > total_samples:
            end_idx = total_samples

        y[start_idx:end_idx] += x

        return y

    else:

        window = sps.windows.tukey(pulse_samples, alpha=0.2)

        # The chirp shape is identical for the direct arrival and every echo; only its phase
        # and amplitude change. Precompute the in-phase and quadrature chirps once, so any
        # phase is a cheap linear combination of the two instead of a fresh chirp() per echo.
        chirp0 = sps.chirp(t_pulse, freq0, pulse_duration, freq1, method=type, phi=0)
        chirp90 = sps.chirp(t_pulse, freq0, pulse_duration, freq1, method=type, phi=90)

        def chirp_at(phase_deg):
            pr = np.deg2rad(phase_deg)
            return chirp0 * np.cos(pr) + chirp90 * np.sin(pr)

        x = amp * chirp_at(phase)
        x = (noise_rms*noise + x) * window

        start_idx = int(round(t_start * fs))
        end_idx = start_idx + pulse_samples

        if end_idx > total_samples:
            end_idx = total_samples

        y[start_idx:end_idx] += x

        dr = 5
        r = r - dr

        while end_idx < total_samples:

            t_start_reverb = t_start + (2 * (r+dr)) / c

            snr_amp = snr - TL((freq0+freq1)/2, 2*(r+dr))

            # snr_amp only decreases as the echo travels further, so once it is below the
            # floor every later echo is too, and the loop can stop.
            if snr_amp < SNR_FLOOR_DB:
                break

            amp = noise_rms * 10**(snr_amp/20)

            phase = np.random.uniform(0, 360)
            x = amp * chirp_at(phase)
            noise_echo = np.random.normal(size=pulse_samples)
            x = (noise_echo + x) * window

            start_idx = int(round(t_start_reverb * fs))
            end_idx = start_idx + pulse_samples

            if end_idx > total_samples:
                end_idx = total_samples
                x = x[:end_idx - start_idx]

            y[start_idx:end_idx] += x
            r += dr

        return y


def generate_start_end(n_pulses):

    """
    Generates a set of non-overlapping start and end times for a given number of pulses, all contained
    within a 5-second window. Each pulse interval has a duration between 0.15 and 1.5 seconds and
    a start time drawn uniformly from [0, 4.84).

    ----------

    Parameters:
        n_pulses (int) - number of pulse intervals to generate.

    Returns:
        picked_intervals (ndarray) - 2-D array of shape (number of placed pulses, 2), where each row
                                     contains the start and end time (in seconds) of a pulse, sorted
                                     by start time.
    """

    switch = 0
    n = 0
    count = 0

    picked_intervals = np.zeros((n_pulses, 2))

    for i in range(n_pulses):

        start = np.random.uniform(0, 4.84)
        end = np.random.uniform(start+0.15, start+1.5)
        pl = end - start

        while start+pl > 5:
            end = np.random.uniform(start+0.15, start+1.5)
            pl = end - start

        for j in range(n_pulses):

            if intervals_overlap(start, end, picked_intervals[j][0], picked_intervals[j][1]):
                switch = 1
                break

        while switch == 1 and n < 1000:

            start = np.random.uniform(0, 4.84)
            end = np.random.uniform(start+0.15, start+1.5)
            pl = end - start

            while start+pl > 5:
                end = np.random.uniform(start+0.15, start+1.5)
                pl = end - start

            for k in range(n_pulses):

                if not intervals_overlap(start, end, picked_intervals[k][0], picked_intervals[k][1]):
                    count += 1

                if count == n_pulses:
                    switch = 0

            count = 0
            n += 1

        if n == 1000:
            break

        else:
            picked_intervals[i][0] = start
            picked_intervals[i][1] = end

    picked_intervals.sort(axis=0)

    picked_intervals = picked_intervals[np.any(picked_intervals != 0, axis=1)]

    return picked_intervals


def _noise_example(x_empty):

    """
    Builds a single noise-only example: the dB spectrogram of one background segment together with a
    target sequence containing only the EOS token. Used as one parallel task in generate_train.

    ----------

    Parameters:
        x_empty (ndarray) - 1-D background noise segment.

    Returns:
        (list) - [t, f, zxx] where zxx is the dB magnitude spectrogram of the segment.
        (ndarray) - target sequence containing a single EOS token, shape (1, 8).
    """

    f, t, zxx = sps.stft(x_empty, fs=30000, nperseg=2**11, noverlap=2**10)
    zxx = db(zxx)

    return [t, f, zxx], np.array([EOS()])


def _pulse_example(data, n_segments, seed):

    """
    Builds a single noise+pulse example: a randomly drawn background segment with 1-4 synthetic pulses
    added, returned as the dB spectrogram and the corresponding flat 8-element target rows terminated
    by an EOS token. Used as one parallel task in generate_train; the seed makes each task an
    independent, reproducible random stream.

    ----------

    Parameters:
        data (ndarray) - 2-D array of pre-normalized noise segments.
        n_segments (int) - number of available segments to sample from.
        seed (int) - seed for this task's random draws.

    Returns:
        (list) - [t, f, zxx] where zxx is the dB magnitude spectrogram of the example.
        (ndarray) - target sequence of shape (number of pulses + 1, 8), ending in an EOS token.
    """

    np.random.seed(seed)
    random.seed(seed)

    seg_idx = np.random.randint(0, n_segments)
    x = data[seg_idx].copy()

    y_elem = []

    amp_noise_rms = rms(x)
    snr_rand = np.random.uniform(-10, 100)
    distance = np.random.uniform(10, 50)
    n_pulses = np.random.randint(1, 5)

    intervals = generate_start_end(n_pulses)

    for j in range(intervals.shape[0]):

        pulse_data_array = np.zeros(8)

        f1 = np.random.uniform(1000, 14990)
        f2 = np.random.uniform(f1+100, f1+4000)

        if f2 > 14990:
            f2 = 14990

        pulse_list = ["cw", "lfm", "hfm"]
        pulse_type = random.choice(pulse_list)

        if pulse_type == "cw":
            f2 = f1
            pulse_type = "linear"
            pulse_data_array[0] = 1

        elif pulse_type == "lfm":
            pulse_type = "linear"
            pulse_data_array[1] = 1

        elif pulse_type == "hfm":
            pulse_type = "hyperbolic"
            pulse_data_array[2] = 1

        pulse_data_array[4] = intervals[j][0]
        pulse_data_array[5] = intervals[j][1]
        pulse_data_array[6] = np.float64(f1)
        pulse_data_array[7] = np.float64(f2)

        y_elem.append(pulse_data_array)

        sig = pulse(amp_noise_rms, snr_rand, pulse_type, intervals[j][0], intervals[j][1], 5, f1, f2, 30000, distance)
        x += sig

    y_elem.append(EOS())
    y_elem = np.array(y_elem)

    f, t, zxx = sps.stft(x, fs=30000, nperseg=2**11, noverlap=2**10)
    zxx = db(zxx)

    return [t, f, zxx], y_elem


def generate_train(data, n_pulse_examples):

    """
    Generates training data for a pulse-detection model by injecting synthetic pulses into pre-normalized
    noise segments. Two types of examples are produced: noise-only examples (one per segment in data,
    with an end-of-sequence token as the target) and noise+pulse examples (n_pulse_examples in total,
    each created by sampling a random segment from data and adding 1-4 randomly generated pulses).
    Each pulse is assigned a random type (CW, LFM, or HFM), start and end time, center frequency,
    bandwidth, SNR, and source distance. The STFT of each example is computed and converted to dB.
    Pulse metadata is stored as a flat 8-element array per pulse, structured as
    [cw_flag, lfm_flag, hfm_flag, eos_flag, t_start, t_stop, f1, f2], and each target sequence is
    terminated with an EOS token of the same shape.

    The input data is assumed to already be z-score normalized.

    ----------

    Parameters:
        data (ndarray) - 2-D array of pre-normalized noise segments, with shape (number of segments,
                         samples per segment).
        n_pulse_examples (int) - number of noise+pulse examples to generate. Segments are sampled
                                 with replacement, so this can exceed data.shape[0].

    Returns:
        None
    """

    n_segments = data.shape[0]

    # Independent seed per pulse task so parallel workers do not share a random stream.
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence().spawn(n_pulse_examples)]

    noise_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_noise_example)(data[i]) for i in range(n_segments)
    )

    pulse_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_pulse_example)(data, n_segments, seeds[i]) for i in range(n_pulse_examples)
    )

    X_input = [r[0] for r in noise_results] + [r[0] for r in pulse_results]
    y_target = [r[1] for r in noise_results] + [r[1] for r in pulse_results]

    X_input = np.array(X_input, dtype=object)
    y_target = np.array(y_target, dtype=object)

    np.save("INPUT.npy", X_input)

    np.save("TARGET.npy", y_target)






background_segments = np.load("ocean_noise_normalized.npy")
t0 = time.perf_counter()
generate_train(background_segments, 500-background_segments.shape[0])
print(f"generate_train took {time.perf_counter() - t0:.1f} s")