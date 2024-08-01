import random
from unittest import result
import torch
import torchaudio
from pathlib import Path
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import torch.multiprocessing as tmp
import numpy as np
from torch.utils.data import IterableDataset
from queue import SimpleQueue




class AudioProcessor2:
    def __init__(self, sample_rate, num_workers=None):
        self.sample_rate = sample_rate
        self.num_workers = num_workers  # Allow user to specify the number of processes

    def load_n_resample(self, file, result_queue, target_sample_rate):
        try:
            waveform, sample_rate = torchaudio.load(file)
            if sample_rate != target_sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
            # result_queue.put(waveform)
            # waveform.share_memory_() 
            # print(f"result_queue size: {result_queue.qsize()}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
            return None

    def multi_process_audio_loading(self, files):
        processes = []
        # manager = mp.Manager()
        result_queue = tmp.Queue()
        for file in files:
            process = tmp.Process(target=self.load_n_resample, args=(file,result_queue, self.sample_rate))
            process.start()
            processes.append(process)
        for process in processes:
            # print("Joining process")
            process.join()
        # print("All processes joined")
        
        waveforms = []
        # while not result_queue.empty():
        #     waveform = result_queue.get()
        #     if waveform is not None:
        #         waveforms.append(waveform)
        return waveforms


def waveform_iterator(waveform, sample_rate = 22050, window_length_s=1, sample_gap=0, samples_per_file=10):
    window_length = int(sample_rate * window_length_s)
    end = waveform.shape[1] - (2*window_length + window_length * sample_gap)
    stride = end // samples_per_file
    max_offset_width = stride // 2

    for i in range(samples_per_file):
        raw_start = i * stride
        offset_min = max(0, raw_start - max_offset_width)
        offset_max = min(end, raw_start + max_offset_width)
        start = np.random.randint(offset_min, offset_max)

        wf1_start = start
        wf1_end = start + window_length
        wf2_start = wf1_end + window_length * sample_gap
        wf2_end = wf2_start + window_length
        
        # print(wf1_start, wf1_end, wf2_start, wf2_end)

        wf1 = waveform[:, wf1_start:wf1_end]
        wf2 = waveform[:, wf2_start:wf2_end]

        yield wf1, wf2

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
            future_to_file = {executor.submit(self.load_n_resample, file): file for file in files}
            for future in as_completed(future_to_file):
                result = future.result()
                result.share_memory_()
                if result is not None:
                    waveforms.append(result)
        # print(f"Returning {len(waveforms)}waveforms")
        return waveforms
    
    # def multi_process_audio_loading(self, files):
    #     waveforms = []
    #     with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
    #         for result in executor.map(self.load_n_resample, [str(file) for file in files]):
    #             if result is not None:
    #                 waveforms.append(result)
    #                 # print("Result appended")
    #     return waveforms


class MTGContrastiveDataset(IterableDataset):
    def __init__(self, audio_loader, audio_folder: Path, sample_rate, window_length_s=10, mask_prob=0, samples_per_file=10, folder_whitelist=None, max_files=None, concurrent_files=1, sample_gap=0 ):
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
            
    def print_profile(self):
        print(f"Iteration - {self.counter}")
        print(f'Worker ID - {torch.utils.data.get_worker_info().id}')
        for key, value in self.profile_vars.items():
            print(f"{key}: {value}")

    def __len__(self):
        return min(len(self.all_files), self.max_files) * self.samples_per_file
    
    def get_worker_files(self):
        all_files = self.all_files
        max_files = self.max_files
        
        all_files = all_files[:max_files] if max_files is not None else all_files
        # Randomize the order of files
        np.random.shuffle(all_files)
        return all_files

    def __iter__(self):
        # Multi-thread handling
        all_files = self.get_worker_files()

        start_file_idx = 0
        while start_file_idx < len(all_files):
            cur_files = all_files[start_file_idx:start_file_idx+self.concurrent_files]
            load_audio = time.time()

            waveforms = self.audio_processor.multi_process_audio_loading(cur_files)
            print(f'    Load audio - {time.time() - load_audio}')

            waveform_generators = [self.waveform_iterator(waveform) for waveform in waveforms]
            for _ in range(self.samples_per_file):
                waveforms = []
                for waveform_generator in waveform_generators:
                    wf1, wf2 = next(waveform_generator)
                    waveforms.extend([wf1, wf2])
                # waveformsc = torch.stack(waveforms)
                yield (1, 2), (3, 4), -1
            
            start_file_idx += self.concurrent_files
        self.counter += 1

if __name__ == '__main__':
    
    tmp.set_start_method('spawn', force=True)

    processor = AudioProcessor2(22050)
    
    num_files = 16
    
    mtg_path = Path('mtg')
    
    reference_sample_rate = 22050
    
    train_folders = [
        "00", "01", "02", "03",
        "04", "05", "06", "07",
        "08", "09", "10", "11",
        "12", "13", "14", "15",
        "16", "17", "18", "19"
    ]
    val_folders = [
        "20"
    ]
    
    batch_size = 64
    
    mtg_train_dataset = MTGContrastiveDataset(processor, mtg_path ,  reference_sample_rate, mask_prob=0.8, samples_per_file=15, folder_whitelist=train_folders, concurrent_files=batch_size)

    # train_loader = torch.utils.data.DataLoader(mtg_train_dataset, batch_size=64)

    for _ in range(10):
        next(iter(mtg_train_dataset))
        
    
        