"""
The MTG dataset loading is more complicated. The total size of the dataset is way too large to keep in memory so we need to incrementally load the files
while returning samples. To do this we move away from map style datasets and use iterable datasets instead.
We also know this will be used for contrastive learning so we will always return pairs of samples.
The order of files should not matter so we just go through the files alphabetically.
"""

import profile
import time
import wave
from torch.utils.data import IterableDataset
from pathlib import Path
import torchaudio
import torch
import numpy as np
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import TimeMasking, FrequencyMasking
from torchaudio.transforms import AmplitudeToDB
import librosa
from torchaudio import transforms
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import torch.multiprocessing as mp


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

class ImprovedAudioProcessor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        
    def process_audio(self, file, result_queue):
        try:
            # Load and process audio without PyTorch
            audio, sr = librosa.load(file, sr=None)
            # Perform some processing (e.g., compute mean amplitude)
            mean_amplitude = audio.mean()
            result_queue.put((audio, mean_amplitude))
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    def load_n_resample(self, file, result_queue):
        try:
            waveform, sample_rate = torchaudio.load(file)
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            result_queue.put(waveform)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    def multi_process_audio_loading(self, files):
        processes = []
        with mp.Manager() as manager:
            result_queue = manager.Queue()
            for file in files:
                process = mp.Process(target=self.load_n_resample, args=(file, result_queue))
                process.start()
                processes.append(process)
            
            for process in processes:
                process.join()
            
            waveforms = []
            while not result_queue.empty():
                try:
                    waveform = result_queue.get(timeout=1)
                    waveforms.append(waveform)
                except mp.queues.Empty:
                    break
            
        return waveforms

class AudioProcessor2:
    def __init__(self, sample_rate, num_workers=None):
        self.sample_rate = sample_rate
        self.num_workers = num_workers  # Allow user to specify the number of processes

    def load_n_resample(self, file, result_queue, target_sample_rate):
        try:
            waveform, sample_rate = torchaudio.load(file)
            if sample_rate != target_sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
            result_queue.put(waveform)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            return None

    def multi_process_audio_loading(self, files):
        processes = []
        with mp.Manager() as manager:

        # manager = mp.Manager()
            result_queue = manager.Queue()
            for file in files:
                process = mp.Process(target=self.load_n_resample, args=(file,result_queue, self.sample_rate))
                process.start()
                processes.append(process)
            for process in processes:
                process.join()
            
            waveforms = []
            while not result_queue.empty():
                waveform = result_queue.get(timeout=1)
                if waveform is not None:
                    waveforms.append(waveform)
        return waveforms
            


class AudioProcessor:
    def __init__(self, sample_rate, num_workers=None):
        self.sample_rate = sample_rate
        self.num_workers = num_workers  # Allow user to specify the number of processes

    def load_n_resample(self, file):
        try:
            waveform, sample_rate = torchaudio.load(file)
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            return waveform
        except Exception as e:
            print(f"Error processing {file}: {e}")
            return None

    def multi_process_audio_loading(self, files):
        waveforms = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # files = [str(file) for file in files]
            print(files)
            future_to_file = {executor.submit(self.load_n_resample, str(file)): file for file in files}
            print('files loaded')
            for future in as_completed(future_to_file):
                result = future.result()
                if result is not None:
                    waveforms.append(result)
                    print("Result appended")
        return waveforms


class MTGContrastiveDataset(IterableDataset):
    def __init__(self, audio_loader, audio_folder: Path, sample_rate, window_length_s=10, mask_prob=0, samples_per_file=10, folder_whitelist=None, max_files=None, concurrent_files=1, sample_gap=0 ):
        logging.info(f"Creating MTGContrastiveDataset with audio_folder: {audio_folder}, sample_rate: {sample_rate}, window_length_s: {window_length_s}, mask_prob: {mask_prob}, samples_per_file: {samples_per_file}, folder_whitelist: {folder_whitelist}")
        self.audio_folder = audio_folder
        self.sample_rate = sample_rate
        self.window_length_s = window_length_s
        self.mask_prob = mask_prob
        self.samples_per_file = samples_per_file
        self.max_files = max_files
        self.folder_whitelist = folder_whitelist
        self.concurrent_files = concurrent_files
        self.sample_gap = sample_gap
        self.audio_processor = audio_loader

        self.all_files = self.get_files()
        if self.max_files is None:
            self.max_files = len(self.all_files)
        
        self.profile_vars = {}
        self.counter = 0

    # def load_audio(self, file):
    #     # load_start = time.time()
    #     waveform, sample_rate = torchaudio.load(file)
    #     # print(f"Load audio: {time.time() - load_start}")
    #     print(f'Sample rate: {sample_rate}')
    #     if sample_rate != self.sample_rate:
    #         # resampling_time = time.time()
    #         # print(f'Resampling {file} from {sample_rate} to {self.sample_rate} in {time.time() - resampling_time}')
    #         waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
    #     return waveform

    # def multithreded_load_audio(self, files):
    #     with ProcessPoolExecutor() as executor:
    #         waveforms = executor.map(self.load_audio, files)
    #     return waveforms

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

    def waveform_index_iterator(self, waveform_shape):
        window_length = int(self.sample_rate * self.window_length_s)
        end = waveform_shape - (2*window_length + window_length * self.sample_gap)
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

            wf1_idx = (wf1_start,wf1_end)
            wf2_idx =(wf2_start,wf2_end)

            yield wf1_idx, wf2_idx


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
        for folder in sorted(self.audio_folder.glob('[!.]*')):
            folder_name = folder.name
            if self.folder_whitelist is not None and folder_name not in self.folder_whitelist:
                continue
            folder_files = sorted(list(folder.glob('[!.]*.mp3')))
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
    
    def print_profile(self):
        print(f"Iteration - {self.counter}")
        print(f'Worker ID - {torch.utils.data.get_worker_info().id}')
        for key, value in self.profile_vars.items():
            print(f"{key}: {value}")

    def __len__(self):
        return min(len(self.all_files), self.max_files) * self.samples_per_file

    def __iter__(self):
        # Multi-thread handling
        # print(f"Going through iter - [{torch.utils.data.get_worker_info().id}]")
        all_files = self.get_worker_files()

        start_file_idx = 0
        start_file_idx_counter = 0
        while start_file_idx < len(all_files):
            # print(f"     Going through while (start_file_idx < len(all_files) -[{torch.utils.data.get_worker_info().id}]")
            all_files_start = time.time()
            cur_files = all_files[start_file_idx:start_file_idx+self.concurrent_files]
            self.profile_vars = { 'init_start': str(self.counter) +'_'+str(time.time() - all_files_start)}
            # print("     loading audio")
            load_audio = time.time()
            # print(f"Starting Audio Loading")
            # print(f'Len of Curr Files - {len(cur_files)} - [{torch.utils.data.get_worker_info().id}]')
            # waveforms = [self.load_audio(file) for file in cur_files]
            # waveforms = self.multithreded_load_audio(cur_files)
            # print(cur_files)
            # self.audio_processor = AudioProcessor(self.sample_rate)

            waveforms = self.audio_processor.multi_process_audio_loading(cur_files)
            print(f'waveforms len - {len(waveforms)}')
            waveform_lengths = [waveforms[i].shape[1] for i in range(len(waveforms))]
            print(f'waveform average length - {np.mean(waveform_lengths)}')
            print(f'    Load audio - {time.time() - load_audio}')
            # print(f'    Num of loaded waveforms - {len(waveforms)}  -[{torch.utils.data.get_worker_info().id}]')
            # self.profile_vars['load_audio'] = str(time.time() - load_audio)
            # waveform_start = time.time()
            waveform_generators = [self.waveform_iterator(waveform) for waveform in waveforms]
            # self.profile_vars['waveform_start'] = str(time.time() - waveform_start)
            first_for = 0
            for _ in range(self.samples_per_file):
            #     print(f"         Going through for (batch in range(self.samples_per_file)")
                waveforms = []
                for waveform_generator in waveform_generators:
            #         # print("             Going through for (waveform_generator in waveform_generators)")
                    wf1, wf2 = next(waveform_generator)
                    waveforms.extend([wf1, wf2])
                    # print(f"waveform len - ")
            #         waveform_generator_start = time.time()
                # waveforms_clone = [waveforms[0].shape]*len(waveforms)
                # for i in range(len(waveforms)):
                #     waveforms_clone[i] = torch.clone(waveforms[i])
                waveforms = torch.stack(waveforms)
                # get_mel_spectrogram_start = time.time()
                mels = get_mel_spectrogram(waveforms, self.sample_rate)
                
            #     self.profile_vars['get_mel_spectrogram'] = str(time.time() - get_mel_spectrogram_start)
                
                # augment_start = time.time()                
                # if self.mask_prob > 0:
                #     for i in range(len(mels)):
                #         if np.random.rand() < self.mask_prob:
                #             pass
                            # mels[i] = self.spectro_augment(mels[i])
            #     self.profile_vars['augment'] = str(time.time() - augment_start)
                
            #     # self.print_profile()
                # for i in range(0, len(mels), 2):
                #     pass
                    # print("             Yielding")
                yield (waveforms, 2), (3, 4), -1
            #     first_for += 1
            
            # print(f"         {first_for} - [{torch.utils.data.get_worker_info().id}]")
                

            start_file_idx += self.concurrent_files
            start_file_idx_counter += 1
            # print(f"     {start_file_idx_counter} - [{torch.utils.data.get_worker_info().id}]")
            # print(f"     All files - {len(all_files)} - [{torch.utils.data.get_worker_info().id}]")
            # print(f"     Start file idx - {start_file_idx} - [{torch.utils.data.get_worker_info().id}]")
        self.counter += 1
        

if __name__ == '__main__':
    print("Starting")
