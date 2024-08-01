"""
Defines utility functions for extracting features from audio files
"""

import librosa
from torchaudio import transforms

def get_mel_spectrogram(waveform, sample_rate, n_mels=64, n_fft=1024, hop_len=None):
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(waveform)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec

def get_zero_crossing_rate(waveform, sample_rate):
    return librosa.feature.zero_crossing_rate(y=waveform.numpy().flatten(), frame_length=1024, hop_length=512)[0]

def get_zero_crossing_rate_stats(waveform, sample_rate):
    zcr = get_zero_crossing_rate(waveform, sample_rate)
    return (zcr.mean(), zcr.std())

def get_spectral_centroid(waveform, sample_rate):
    return librosa.feature.spectral_centroid(y=waveform.numpy().flatten(), sr=sample_rate)[0]

def get_spectral_centroid_stats(waveform, sample_rate):
    centroid = get_spectral_centroid(waveform, sample_rate)
    return (centroid.mean(), centroid.std())

def get_spectral_rolloff(waveform, sample_rate):
    return librosa.feature.spectral_rolloff(y=waveform.numpy().flatten(), sr=sample_rate)[0]

def get_spectral_rolloff_stats(waveform, sample_rate):
    rolloff = get_spectral_rolloff(waveform, sample_rate)
    return (rolloff.mean(), rolloff.std())

def get_spectral_bandwidth(waveform, sample_rate):
    return librosa.feature.spectral_bandwidth(y=waveform.numpy().flatten(), sr=sample_rate)[0]

def get_spectral_bandwidth_stats(waveform, sample_rate):
    bandwidth = get_spectral_bandwidth(waveform, sample_rate)
    return (bandwidth.mean(), bandwidth.std())

def get_spectral_contrast(waveform, sample_rate):
    return librosa.feature.spectral_contrast(y=waveform.numpy().flatten(), sr=sample_rate)[0]

def get_spectral_contrast_stats(waveform, sample_rate):
    contrast = get_spectral_contrast(waveform, sample_rate)
    return (contrast.mean(), contrast.std())