"""
Utilities for visualizing the data
"""

import matplotlib.pyplot as plt
import numpy as np

def visualize_mel_spectrogram(mel_spectrogram, width=10, show_axes=True):
    fig = plt.figure(figsize=(width, 4))
    ax = fig.add_subplot(111)
    ax.imshow(mel_spectrogram[0], aspect='auto', origin='lower', cmap='viridis')

    if not show_axes:
        ax.set_axis_off()

    plt.show()
    plt.close()

def visualize_waveform(waveform, sample_rate, width=10, show_axes=True):
    fig = plt.figure(figsize=(width, 4))
    ax = fig.add_subplot(111)

    num_channels, num_frames = waveform.shape
    time_axis = np.linspace(0, num_frames / sample_rate, num_frames)

    if num_channels == 1:
        ax.plot(time_axis, waveform[0], linewidth=1)
    else:
        for c in range(num_channels):
            ax.plot(time_axis, waveform[c], linewidth=1)
            ax.grid(True)

    if not show_axes:
        ax.set_axis_off()

    plt.show()
    plt.close()