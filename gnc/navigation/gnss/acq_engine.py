import copy
import timeit

import numpy as np
from matplotlib import pyplot as plt

from signals.waveform import NavigationWaveform, Waveform


def acq_iq_example(coherent_integration_blocks, block_time, css, css_steps, fss, fss_steps):
    BLOCK_TIME = block_time
    BLOCKS = coherent_integration_blocks
    f_ref = 1.023E6
    chipping_freq = f_ref*1
    carrier_freq = 0.0

    sample_time = BLOCK_TIME
    sample_freq = 4.092E6 / 2
    samples = int(sample_time*sample_freq)

    chipping_rate = 1/chipping_freq
    chip_scan = np.linspace(css[0], css[1], css_steps, endpoint=True)
    doppler_scan = np.linspace(fss[0], fss[1], fss_steps, endpoint=True)
    correlation = np.zeros([chip_scan.shape[0], doppler_scan.shape[0]], dtype=np.complex128)

    for block in range(BLOCKS):
        print('Block: %s of %s' % (block,BLOCKS))
        start_time = timeit.default_timer()
        block_start = block * BLOCK_TIME
        signal = NavigationWaveform.bpsk(samples=samples, sampling_freq=sample_freq,
                                         carrier_freq=carrier_freq, code_type='CA', prn=1, gps_time_start=1000.0+block_start)
        signal.adjust_power(-160.0)
        noise = Waveform.awgn(time=sample_time, sampling_frequency=sample_freq, center_freq=0E6, bw=20E6,
                                      power=-140.0)
        for chip_ind, chip in enumerate(chip_scan):
            replica_offset = chip * chipping_rate
            replica_signal = NavigationWaveform.bpsk(samples=samples, sampling_freq=sample_freq,
                                             carrier_freq=carrier_freq, code_type='CA', prn=1, gps_time_start=1000.0 + replica_offset + block_start)
            for dop_ind, doppler in enumerate(doppler_scan):
                composite_signal = copy.deepcopy(signal) + copy.deepcopy(noise)
                composite_signal.freq_shift(doppler)
                correlation[chip_ind, dop_ind] += composite_signal.coherent_correlation(replica_signal)

        print('Block time: %s' % (timeit.default_timer() - start_time))
    correlation = 10.0*np.log10(correlation)

    chip_mesh, dop_mesh = np.meshgrid(chip_scan, doppler_scan)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if doppler_scan.shape[0] > 1:
        surf = ax.plot_surface(chip_mesh, dop_mesh, correlation.T, cmap="viridis")
        plt.xlabel('Code Offset (Chips)')
        plt.ylabel('Frequency Offset (Hz)')
        plt.colorbar(surf)
    else:
        plt.plot(chip_scan, correlation[:,0])
        plt.xlabel('Code Offset (Chips)')
        plt.ylabel('Correlation')
        plt.grid(True)
    plt.show()