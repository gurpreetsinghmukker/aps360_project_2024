import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torchaudio import transforms

from transform_utilities import get_mel_spectrogram

class GTZANDataset(Dataset):
    """

    """
    def __init__(self, audio_tensors, genres, genre_index_map, sample_rate, window_length_s = 10, mask_prob = 0, contrastive=False, randomize=True, start_prop=0):
        """
        window_length_s: The length of the window in seconds
        """
        self.audio_tensors = audio_tensors
        self.genres = genres
        self.genre_index_map = genre_index_map
        self.sample_rate = sample_rate
        self.window_length_s = window_length_s
        self.mask_prob = mask_prob
        self.contrastive = contrastive
        self.randomize = randomize
        self.start_prop = start_prop

        self.num_volume_augmented = None

    def idx_to_sample(self, idx):
        for i, genre in enumerate(self.genres):
            if idx < len(self.audio_tensors[i]):
                return i, idx
            idx -= len(self.audio_tensors[i])

        return None

    def spectro_augment(self, mel, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = mel.shape
        mask_value = mel.mean()
        aug_mel = mel

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_mel = transforms.FrequencyMasking(freq_mask_param)(aug_mel, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_mel = transforms.TimeMasking(time_mask_param)(aug_mel, mask_value)

        return aug_mel

    def __len__(self):
        return sum(len(genre) for genre in self.audio_tensors.values())

    def process_waveform(self, waveform):
        mel_spectrogram = get_mel_spectrogram(waveform, self.sample_rate)
        if self.mask_prob > 0:
            if np.random.rand() < self.mask_prob:
                mel_spectrogram = self.spectro_augment(mel_spectrogram)

        return waveform, mel_spectrogram

    def __getitem__(self, idx):
        genre, idx = self.idx_to_sample(idx)
        waveform = self.audio_tensors[genre][idx]
        if self.contrastive:
            # If this is the case, we double the window length to get two adjacent waveforms
            window_length = int(self.sample_rate * self.window_length_s*2)
        else:
            window_length = int(self.sample_rate * self.window_length_s)
        if self.randomize:
            # This sampling scheme biases us toward the center of the audio
            start = np.random.randint(0, waveform.shape[1] - window_length)
            # start = 0
            end = start + window_length
        else:
            start = int((waveform.shape[1] - window_length) * self.start_prop)
            end = start + window_length
        waveform = waveform[:, start:end]

        genre = self.genre_index_map[self.genres[genre]]
        if self.contrastive:
            # Then we split the one waveform in two
            waveform1 = waveform[:, :(window_length//2)]
            waveform2 = waveform[:, (window_length//2):]

            waveform1, mel_spectrogram1 = self.process_waveform(waveform1)
            waveform2, mel_spectrogram2 = self.process_waveform(waveform2)

            return (waveform1, waveform2), (mel_spectrogram1, mel_spectrogram2), genre
        else:
            waveform, mel_spectrogram = self.process_waveform(waveform)
            return waveform, mel_spectrogram, genre

def gtzan_generate_folds(audio_tensors, genres, reference_sample_rate, k_folds=5, mask_prob=0.0, contrastive=False):
    """
    Defines a generator that yields train and val datasets for each fold.
    """
    # First, we get a single list that includes all tensors
    X = []
    y = []
    for i, genre in enumerate(genres):
        X += audio_tensors[i]
        y += [i] * len(audio_tensors[i])

    genre_index_map = {genre: i for i, genre in enumerate(genres)}

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(X, y):
        # Here we must reconstruct an audio_tensors dict for each set
        train_audio_tensors = {
            genre_index_map[genre]: [] for genre in genres
        }
        val_audio_tensors = {
            genre_index_map[genre]: [] for genre in genres
        }
        for train_idx in train_index:
            input = X[train_idx]
            genre = y[train_idx]
            train_audio_tensors[genre].append(input)

        for val_idx in val_index:
            input = X[val_idx]
            genre = y[val_idx]
            val_audio_tensors[genre].append(input)


        train_dataset = GTZANDataset(
            train_audio_tensors,
            genres,
            genre_index_map,
            reference_sample_rate,
            mask_prob=mask_prob,
            contrastive=contrastive
        )
        val_dataset = GTZANDataset(val_audio_tensors, genres, genre_index_map, reference_sample_rate, contrastive=contrastive)
        yield train_dataset, val_dataset
