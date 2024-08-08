"""
The MTG dataset loading is more complicated. The total size of the dataset is way too large to keep in memory so we need to incrementally load the files
while returning samples. To do this we move away from map style datasets and use iterable datasets instead.
We also know this will be used for contrastive learning so we will always return pairs of samples.
The order of files should not matter so we just go through the files alphabetically.
"""

from torch.utils.data import IterableDataset
from pathlib import Path
import torchaudio
import torch
import numpy as np
from torchaudio import transforms
import logging

# Setup logging
logging.basicConfig(filename='dataset.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Started')

def worker_init_fn(worker_id):
    log_handler = logging.FileHandler('dataset.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    logger.info(f"Worker {worker_id} initializing.")

def get_mel_spectrogram(waveform, sample_rate, n_mels=64, n_fft=1024, hop_len=None):
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(waveform)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec

class MTG_Mel_ContrastiveDataset(IterableDataset):
    def __init__(self, mel_spect_folder: Path, sample_rate, window_length_s=10, mask_prob=0, samples_per_file=10, folder_whitelist=None, max_files=None, concurrent_files=1, sample_gap=0):
        print("Using MTGContrastiveDataset")
        logging.info(f"Creating MTGContrastiveDataset with audio_folder: {mel_spect_folder}, sample_rate: {sample_rate}, window_length_s: {window_length_s}, mask_prob: {mask_prob}, samples_per_file: {samples_per_file}, folder_whitelist: {folder_whitelist}")
        self.mel_spect_folder = mel_spect_folder
        self.sample_rate = sample_rate
        self.window_length_s = window_length_s
        self.mask_prob = mask_prob
        self.samples_per_file = samples_per_file
        self.max_files = max_files
        self.folder_whitelist = folder_whitelist
        self.concurrent_files = concurrent_files
        self.sample_gap = sample_gap

        self.all_files = self.get_files()
        if self.max_files is None:
            self.max_files = len(self.all_files)

    def load_audio(self, file):
        waveform, sample_rate = torchaudio.load(file)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform

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

    def process_waveform(self, waveform):
        mel_spectrogram = get_mel_spectrogram(waveform, self.sample_rate)
        if self.mask_prob > 0:
            if np.random.rand() < self.mask_prob:
                mel_spectrogram = self.spectro_augment(mel_spectrogram)

        return waveform, mel_spectrogram

    def mel_iterator(self, mel, hop_length = 512, mel_sample_gap = 0):
        mel_window_length = int((self.sample_rate * self.window_length_s) // hop_length +1)
        end = mel.shape[2] - (2*mel_window_length + mel_window_length * mel_sample_gap)
        stride = end // self.samples_per_file
        max_offset_width = stride // 2

        for i in range(self.samples_per_file):
            raw_start = i * stride
            offset_min = max(0, raw_start - max_offset_width)
            offset_max = min(end, raw_start + max_offset_width)
            start = np.random.randint(offset_min, offset_max)
            mel1_start = start
            mel1_end = start + mel_window_length
            mel2_start = mel1_end + mel_window_length * self.sample_gap
            mel2_end = mel2_start + mel_window_length

            mel1 = mel[:,:, mel1_start:mel1_end]
            mel2 = mel[:,:, mel2_start:mel2_end]

            yield mel1, mel2

    def waveform_iterator(self, waveform):
        window_length = int(self.sample_rate * self.window_length_s)
        end = waveform.shape[1] - (2*window_length + window_length * self.sample_gap)
        stride = end // self.samples_per_file
        max_offset_width = stride // 2

        for i in range(self.samples_per_file):
            raw_start = i * stride
            offset_min = max(0, raw_start - max_offset_width)
            offset_max = min(end, raw_start + max_offset_width)
            start = np.random.randint(offset_min, offset_max)

            wf1_start = start
            wf1_end = start + window_length
            wf2_start = wf1_end + window_length * self.sample_gap
            wf2_end = wf2_start + window_length

            wf1 = waveform[:, wf1_start:wf1_end]
            wf2 = waveform[:, wf2_start:wf2_end]

            yield wf1, wf2

    def get_files(self):
        # Returns a list of all files that will be used
        files = []
        for folder in sorted(self.mel_spect_folder.glob('[!.]*')):
            folder_name = folder.name
            if self.folder_whitelist is not None and folder_name not in self.folder_whitelist:
                continue
            folder_files = sorted(list(folder.glob('[!.]*.pt')))
            for file in folder_files:
                files.append(file)
        return files

    def get_worker_files(self):
        worker_info = torch.utils.data.get_worker_info()
        logging.info(f"Worker info: {worker_info}")
        all_files = self.all_files
        max_files = self.max_files
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            max_files = self.max_files // num_workers if self.max_files is not None else None

            split_idxs = np.array_split(np.arange(len(self.all_files)), num_workers)
            cur_idxs = split_idxs[worker_id]

            all_files = [self.all_files[i] for i in cur_idxs]
        else:
            worker_id = -1
        all_files = all_files[:max_files] if max_files is not None else all_files
        logging.info(f"Worker {worker_id} has {len(all_files)} files (Max files: {max_files})")
        # Randomize the order of files
        np.random.shuffle(all_files)
        return all_files

    def __len__(self):
        return min(len(self.all_files), self.max_files) * self.samples_per_file

    def __iter__(self):
        all_files = self.get_worker_files()

        start_file_idx = 0
        while start_file_idx < len(all_files):
            cur_files = all_files[start_file_idx:start_file_idx+self.concurrent_files]
            mel_spects = [torch.load(file, weights_only= True) for file in cur_files]
            mel_spects_generators = [self.mel_iterator(mel_spect, ) for mel_spect in mel_spects]

            for batch in range(self.samples_per_file):
                mels = []
                for mel_spects_generator in mel_spects_generators:
                    mel1, mel2 = next(mel_spects_generator)
                    mels.extend([mel1, mel2])
                mels = torch.stack(mels)
                if self.mask_prob > 0:
                    for i in range(len(mels)):
                        if np.random.rand() < self.mask_prob:
                            mels[i] = self.spectro_augment(mels[i])
                for i in range(0, len(mels), 2):
                    yield ( 1, 1), (mels[i], mels[i+1]), -1
            start_file_idx += self.concurrent_files
            
class MTG_Mel_ContrastiveDataset2(IterableDataset):
    def __init__(self, mel_spect_folder: Path, sample_rate, window_length_s=10, mask_prob=0, samples_per_file=10, folder_whitelist=None, max_files=None, concurrent_files=1, sample_gap=0):
        print("Using MTG_Mel_ContrastiveDataset2")
        logging.info(f"Creating MTGContrastiveDataset2 with audio_folder: {mel_spect_folder}, sample_rate: {sample_rate}, window_length_s: {window_length_s}, mask_prob: {mask_prob}, samples_per_file: {samples_per_file}, folder_whitelist: {folder_whitelist}")
        self.mel_spect_folder = mel_spect_folder
        self.sample_rate = sample_rate
        self.window_length_s = window_length_s
        self.mask_prob = mask_prob
        self.samples_per_file = samples_per_file
        self.max_files = max_files
        self.folder_whitelist = folder_whitelist
        self.concurrent_files = concurrent_files
        self.sample_gap = sample_gap

        self.all_files = self.get_files()
        if self.max_files is None:
            self.max_files = len(self.all_files)

    def load_audio(self, file):
        waveform, sample_rate = torchaudio.load(file)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform

    def spectro_augment(self, mel, max_mask_pct=0.2, n_freq_masks=1, n_time_masks=1, stretch=1.2):
        _, n_mels, n_steps = mel.shape
        mask_value = mel.mean()
        aug_mel = mel

        # aug_mel_2 = transforms.TimeStretch()(aug_mel, stretch)
        # print(aug_mel_2.shape)
    
        freq_mask_param = max_mask_pct * n_mels 
        for _ in range(n_freq_masks):
            aug_mel = transforms.FrequencyMasking(freq_mask_param)(aug_mel, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_mel = transforms.TimeMasking(time_mask_param)(aug_mel, mask_value)

        return aug_mel

    def process_waveform(self, waveform):
        mel_spectrogram = get_mel_spectrogram(waveform, self.sample_rate)
        if self.mask_prob > 0:
            if np.random.rand() < self.mask_prob:
                mel_spectrogram = self.spectro_augment(mel_spectrogram)

        return waveform, mel_spectrogram

            
    def mel_iterator(self, mel, hop_length = 512, mel_sample_gap = 0):
        mel_window_length = int((self.sample_rate * self.window_length_s) // hop_length +1)
        end = mel.shape[2] - (mel_window_length + mel_window_length * mel_sample_gap)
        stride = end // self.samples_per_file
        # max_offset_width = stride // 2

        for i in range(self.samples_per_file):
            start = i * stride
            end = start + mel_window_length
            mel_sample = mel[:,:, start:end]
            yield mel_sample

    def waveform_iterator(self, waveform):
        window_length = int(self.sample_rate * self.window_length_s)
        end = waveform.shape[1] - (2*window_length + window_length * self.sample_gap)
        stride = end // self.samples_per_file
        max_offset_width = stride // 2

        for i in range(self.samples_per_file):
            raw_start = i * stride
            offset_min = max(0, raw_start - max_offset_width)
            offset_max = min(end, raw_start + max_offset_width)
            start = np.random.randint(offset_min, offset_max)

            wf1_start = start
            wf1_end = start + window_length
            wf2_start = wf1_end + window_length * self.sample_gap
            wf2_end = wf2_start + window_length

            wf1 = waveform[:, wf1_start:wf1_end]
            wf2 = waveform[:, wf2_start:wf2_end]

            yield wf1, wf2

    def get_files(self):
        # Returns a list of all files that will be used
        files = []
        for folder in sorted(self.mel_spect_folder.glob('[!.]*')):
            folder_name = folder.name
            if self.folder_whitelist is not None and folder_name not in self.folder_whitelist:
                continue
            folder_files = sorted(list(folder.glob('[!.]*.pt')))
            for file in folder_files:
                files.append(file)
        return files

    def get_worker_files(self):
        worker_info = torch.utils.data.get_worker_info()
        logging.info(f"Worker info: {worker_info}")
        all_files = self.all_files
        max_files = self.max_files
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            max_files = self.max_files // num_workers if self.max_files is not None else None

            split_idxs = np.array_split(np.arange(len(self.all_files)), num_workers)
            cur_idxs = split_idxs[worker_id]

            all_files = [self.all_files[i] for i in cur_idxs]
        else:
            worker_id = -1
        all_files = all_files[:max_files] if max_files is not None else all_files
        logging.info(f"Worker {worker_id} has {len(all_files)} files (Max files: {max_files})")
        # Randomize the order of files
        np.random.shuffle(all_files)
        return all_files

    def __len__(self):
        return min(len(self.all_files), self.max_files) * self.samples_per_file

    def __iter__(self):
        all_files = self.get_worker_files()

        start_file_idx = 0
        while start_file_idx < len(all_files):
            cur_files = all_files[start_file_idx:start_file_idx+self.concurrent_files]
            mel_spects = [torch.load(file, weights_only= True) for file in cur_files]
            mel_spects_generators = [self.mel_iterator(mel_spect, ) for mel_spect in mel_spects]

            for batch in range(self.samples_per_file):
                mels = []
                for mel_spects_generator in mel_spects_generators:
                    mel = next(mel_spects_generator)
                    mels.extend([mel])
                mels = torch.stack(mels)
                mels_copy = mels.clone()
                if self.mask_prob > 0:
                    for i in range(len(mels)):
                        if np.random.rand() < self.mask_prob:
                            mels[i] = self.spectro_augment(mels[i])
                            mels_copy[i] = self.spectro_augment(mels_copy[i])
                for i in range(0, len(mels)):
                    yield (1, 1), (mels[i], mels_copy[i]), -1
            start_file_idx += self.concurrent_files
            
class MTG_Mel_ContrastiveDataset3(IterableDataset):
    def __init__(self, mel_spect_folder: Path, sample_rate, window_length_s=10, mask_prob=0, samples_per_file=10, folder_whitelist=None, max_files=None, concurrent_files=1, sample_gap=0):
        print("Using MTGContrastiveDataset3")
        logging.info(f"Creating MTGContrastiveDataset3 with audio_folder: {mel_spect_folder}, sample_rate: {sample_rate}, window_length_s: {window_length_s}, mask_prob: {mask_prob}, samples_per_file: {samples_per_file}, folder_whitelist: {folder_whitelist}")
        self.mel_spect_folder = mel_spect_folder
        self.sample_rate = sample_rate
        self.window_length_s = window_length_s
        self.mask_prob = mask_prob
        self.samples_per_file = samples_per_file
        self.max_files = max_files
        self.folder_whitelist = folder_whitelist
        self.concurrent_files = concurrent_files
        self.sample_gap = sample_gap

        self.all_files = self.get_files()
        if self.max_files is None:
            self.max_files = len(self.all_files)

    def load_audio(self, file):
        waveform, sample_rate = torchaudio.load(file)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform

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

    def process_waveform(self, waveform):
        mel_spectrogram = get_mel_spectrogram(waveform, self.sample_rate)
        if self.mask_prob > 0:
            if np.random.rand() < self.mask_prob:
                mel_spectrogram = self.spectro_augment(mel_spectrogram)

        return waveform, mel_spectrogram

    def mel_iterator(self, mel, hop_length = 512, mel_sample_gap = 0):
        mel_window_length = int((self.sample_rate * self.window_length_s) // hop_length +1)
        end = mel.shape[2] - (2*mel_window_length + mel_window_length * mel_sample_gap)
        stride = end // self.samples_per_file
        max_offset_width = stride // 2

        for i in range(self.samples_per_file):
            raw_start = i * stride
            offset_min = max(0, raw_start - max_offset_width)
            offset_max = min(end, raw_start + max_offset_width)
            start = np.random.randint(offset_min, offset_max)
            mel1_start = start
            mel1_end = start + mel_window_length
            overlap = np.random.uniform(0.1, 0.5)
            mel2_start = mel1_end - int(mel_window_length*overlap) + mel_window_length * self.sample_gap
            mel2_end = mel2_start + mel_window_length

            mel1 = mel[:,:, mel1_start:mel1_end]
            mel2 = mel[:,:, mel2_start:mel2_end]

            yield mel1, mel2

    def waveform_iterator(self, waveform):
        window_length = int(self.sample_rate * self.window_length_s)
        end = waveform.shape[1] - (2*window_length + window_length * self.sample_gap)
        stride = end // self.samples_per_file
        max_offset_width = stride // 2

        for i in range(self.samples_per_file):
            raw_start = i * stride
            offset_min = max(0, raw_start - max_offset_width)
            offset_max = min(end, raw_start + max_offset_width)
            start = np.random.randint(offset_min, offset_max)

            wf1_start = start
            wf1_end = start + window_length
            wf2_start = wf1_end + window_length * self.sample_gap
            wf2_end = wf2_start + window_length

            wf1 = waveform[:, wf1_start:wf1_end]
            wf2 = waveform[:, wf2_start:wf2_end]

            yield wf1, wf2

    def get_files(self):
        # Returns a list of all files that will be used
        files = []
        for folder in sorted(self.mel_spect_folder.glob('[!.]*')):
            folder_name = folder.name
            if self.folder_whitelist is not None and folder_name not in self.folder_whitelist:
                continue
            folder_files = sorted(list(folder.glob('[!.]*.pt')))
            for file in folder_files:
                files.append(file)
        return files

    def get_worker_files(self):
        worker_info = torch.utils.data.get_worker_info()
        logging.info(f"Worker info: {worker_info}")
        all_files = self.all_files
        max_files = self.max_files
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            max_files = self.max_files // num_workers if self.max_files is not None else None

            split_idxs = np.array_split(np.arange(len(self.all_files)), num_workers)
            cur_idxs = split_idxs[worker_id]

            all_files = [self.all_files[i] for i in cur_idxs]
        else:
            worker_id = -1
        all_files = all_files[:max_files] if max_files is not None else all_files
        logging.info(f"Worker {worker_id} has {len(all_files)} files (Max files: {max_files})")
        # Randomize the order of files
        np.random.shuffle(all_files)
        return all_files

    def __len__(self):
        return min(len(self.all_files), self.max_files) * self.samples_per_file

    def __iter__(self):
        all_files = self.get_worker_files()

        start_file_idx = 0
        while start_file_idx < len(all_files):
            cur_files = all_files[start_file_idx:start_file_idx+self.concurrent_files]
            mel_spects = [torch.load(file, weights_only= True) for file in cur_files]
            mel_spects_generators = [self.mel_iterator(mel_spect, ) for mel_spect in mel_spects]

            for batch in range(self.samples_per_file):
                mels = []
                for mel_spects_generator in mel_spects_generators:
                    mel1, mel2 = next(mel_spects_generator)
                    mels.extend([mel1, mel2])
                mels = torch.stack(mels)
                if self.mask_prob > 0:
                    for i in range(len(mels)):
                        if np.random.rand() < self.mask_prob:
                            mels[i] = self.spectro_augment(mels[i])
                for i in range(0, len(mels), 2):
                    yield ( 1, 1), (mels[i], mels[i+1]), -1
            start_file_idx += self.concurrent_files