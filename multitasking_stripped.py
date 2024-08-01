from unittest import result
import wave
import torch
from pathlib import Path
# import torch.multiprocessing as tmp
import multiprocessing as tmp
import numpy as np
from torch.utils.data import IterableDataset
import torchaudio
import time
import librosa
import pydub
from torchaudio import transforms

class ImprovedAudioProcessor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        
    def process_audio(self, file, result_queue):
        try:
            # Load and process audio without PyTorch
            audio = pydub.AudioSegment.from_mp3(file).set_frame_rate(self.sample_rate)
            # Perform some processing (e.g., compute mean amplitude)
            a = np.array(audio.get_array_of_samples()).reshape(1,-1)
            print(a.shape)
            result_queue.put(a)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    def load_n_resample(self, file, result_queue):
        try:
            waveform, sample_rate = torchaudio.load(file)
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
                print(waveform.shape)
            result_queue.put(waveform)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    def multi_process_audio_loading(self, files):
        with tmp.Manager() as manager:
            processes = []

            result_queue = manager.Queue()
            for file in files:
                process = tmp.Process(target=self.load_n_resample, args=(file, result_queue))
                process.start()
                processes.append(process)
            
            for process in processes:
                process.join()
            
            waveforms = []
            while not result_queue.empty():
                try:
                    waveform = result_queue.get(timeout=1)
                    waveforms.append(waveform)
                except tmp.queues.Empty:
                    break
            
        return waveforms



def load_n_resample(file, result_queue, target_sample_rate):
    try:
        waveform, sample_rate = torchaudio.load(file)
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        result_queue.put(waveform)
        waveform.share_memory_() 
        # print(f"result_queue size: {result_queue.qsize()}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

class AudioProcessor2:
    def __init__(self, resample_rate):
        self.target_resample_rate = resample_rate        
    def multi_process_audio_loading(self, files):
        with tmp.Pool(processes=len(files)) as pool:
            result_queue = tmp.Manager().Queue()
            
            pool.starmap(load_n_resample, [(file, result_queue, self.target_resample_rate) for file in files ])
            waveforms = []
            while not result_queue.empty():
                try:
                    waveform = result_queue.get(timeout=1)
                    waveforms.append(waveform)
                except tmp.queues.Empty:
                    break
                
        # print(f"Returning {len(waveforms)} waveforms")
        return waveforms

# def load_n_resample(file, result_queue, target_sample_rate):
#     try:
#         waveform, sample_rate = torchaudio.load(file)
#         if sample_rate != target_sample_rate:
#             waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
#         # result_queue.put(waveform)
#         # waveform.share_memory_() 
#         # print(f"result_queue size: {result_queue.qsize()}")
#     except Exception as e:
#         print(f"Error processing {file}: {e}")
#         return None


# def load_n_resample(result_queue):
#     tensor = torch.rand((10,10))
#     tensor.share_memory_()
#     result_queue.put(tensor)
    

# class AudioProcessor2:
#     def __init__(self, sample_rate):
#         self.sample_rate = sample_rate        
#     def multi_process_audio_loading(self, files):
#         processes = []
#         result_queue = tmp.Manager().Queue()
        
#         for _, file in enumerate(files, 0):
#             process = tmp.Process(target=load_n_resample, args=(file, result_queue,self.sample_rate))
#             process.start()
#             processes.append(process)
#         for process in processes:
#             process.join()
#         print("All processes joined")
#         waveforms = []
            
#         # while not result_queue.empty():
#         #     print("Getting waveform from queue")
#         #     waveform = result_queue.get()
#         #     print("Got waveform from queue")
#         #     if waveform is not None:
#         #         waveforms.append(waveform)
#         print(f"Returning {len(waveforms)}waveforms")
#         return waveforms

def get_mel_spectrogram(waveform, sample_rate, n_mels=64, n_fft=1024, hop_len=None):
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(waveform)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec

class MTGContrastiveDataset(IterableDataset):
    def __init__(self, sample_rate=22050, window_length_s=1, sample_gap=0.5, samples_per_file=10):
        self.sample_rate = sample_rate
        self.window_length_s = window_length_s
        self.sample_gap = sample_gap
        self.samples_per_file = samples_per_file
    
    
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
            
    def __iter__(self):
        processor = ImprovedAudioProcessor(self.sample_rate)
        path = Path("mtg/01")
        files = list(path.glob("*.mp3"))
        # files = [str(file) for file in files]
        num_files = len(files)
        
        chunk_size = 1
        
        num_chunks = num_files // chunk_size
        last_chunk_size = num_files % chunk_size
        list_of_chunks = [files[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
        list_of_chunks.append(files[-last_chunk_size:])
        # print(list_of_chunks[0])
        # print(list_of_chunks[1])
        
        for chunk in list_of_chunks:
            print(chunk)
            start_time = time.time()
            waveforms = processor.multi_process_audio_loading(chunk)
            waveform_lengths = [waveforms[i].shape[1] for i in range(len(waveforms))]
            wavform_mels = [get_mel_spectrogram(waveform, self.sample_rate) for waveform in waveforms]
            
            print(f'waveform average length - {np.mean(waveform_lengths)}')
            print(f"Time taken: {time.time() - start_time}")
            
            waveform_generators = [self.waveform_iterator(torch.tensor(waveform)) for waveform in waveforms]
            for _ in range(self.samples_per_file):
                waveforms = []
                for waveform_generator in waveform_generators:
                    wf1, wf2 = next(waveform_generator)
                    waveforms.extend([wf1, wf2])
                # waveforms = torch.stack(waveforms)
                yield waveforms
            
if __name__ == '__main__':
    
    tmp.set_start_method('spawn', force=True)
    mtg_train_dataset = MTGContrastiveDataset(sample_rate=22050, window_length_s=10, sample_gap=0, samples_per_file=10)
    # mtg_dataloader = torch.utils.data.DataLoader(mtg_train_dataset)
    for i in mtg_train_dataset:
        print(torch.stack(i).shape)
        
        
# import torch
# from pathlib import Path
# import torch.multiprocessing as tmp
# import numpy as np
# from torch.utils.data import IterableDataset

# def load_n_resample(result_queue):
#     try:
#         tensor = torch.rand((10,10))
#         result_queue.put(tensor)
#     except Exception as e:
#         print(f"Error in child process: {e}")

# class AudioProcessor2:
#     def __init__(self, num_of_procs):
#         self.num_of_procs = num_of_procs
        
#     def multi_process_audio_loading(self):
#         with tmp.Pool(processes=self.num_of_procs) as pool:
#             result_queue = tmp.Manager().Queue()
            
#             pool.starmap(load_n_resample, [(result_queue,) for _ in range(self.num_of_procs)])
            
#             waveforms = []
#             while not result_queue.empty():
#                 try:
#                     waveform = result_queue.get(timeout=1)
#                     waveforms.append(waveform)
#                 except tmp.queues.Empty:
#                     break
                
#         print(f"Returning {len(waveforms)} waveforms")
#         return waveforms

# class MTGContrastiveDataset(IterableDataset):
#     def __init__(self, iters):
#         self.iters = iters
        
#     def __iter__(self):
#         processor = AudioProcessor2(10)
#         for iter in range(self.iters):
#             return_val = processor.multi_process_audio_loading()
#             yield iter
            
# if __name__ == '__main__':
#     tmp.set_start_method('spawn', force=True)
    
#     mtg_train_dataset = MTGContrastiveDataset(10)

#     for _ in range(10):
#         print(next(iter(mtg_train_dataset)))