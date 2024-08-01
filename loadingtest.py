import wave
import torch
import torchaudio
from torchaudio import transforms
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def copy_folder_structure(src: Path, dest: Path):
    for dir in src.glob('**/'):
        if dir == src:
            continue
        else:
            if dir.is_dir():
                if Path.is_dir(dest / dir.relative_to(src)):
                    continue
                else:
                    Path.mkdir(dest / dir.relative_to(src), exist_ok=True)
        

def save_mel_spec(file, sample_rate,src:Path, dest: Path, n_mels=64, n_fft=1024, hop_len=None):
    dest_file_path = dest / file.relative_to(src).with_suffix('.pt')
    try:
        if Path.exists(dest_file_path):
            return f"File {file} already preocessed"
        waveform, sample_rate = torchaudio.load(file)
        mel_spec = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(waveform)
        torch.save(mel_spec, dest_file_path)
        return f"Saved {file} to {dest_file_path}"
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return False

def mp3_to_mel(src: Path, dest: Path, sample_rate=22050, n_mels=64, n_fft=1024, hop_len=None):
    
    files = list(src.glob('**/*.mp3'))
    procs = []
    with ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(save_mel_spec, file, sample_rate, src, dest, n_mels, n_fft, hop_len): file for file in files}
            for future in as_completed(future_to_file):
                result = future.result()
                if result is not None:
                    print(f"Result: {result}") 

if __name__ == "__main__":

    mtg_path = Path('mtg')
    mtg_mel_path = Path('mtg_mel')
    if not mtg_mel_path.exists():
        mtg_mel_path.mkdir(exist_ok=True)
    
    copy_folder_structure(mtg_path, mtg_mel_path)
    mp3_to_mel(mtg_path, mtg_mel_path)

        
    
    
    