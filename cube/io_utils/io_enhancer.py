import copy

import torch
import numpy as np
import librosa
from os import listdir
from os.path import isfile, join
from scipy.signal import resample
import wave

from torch.utils.data.dataset import Dataset
import sys

sys.path.append('')
from cube.io_utils.audio import alter


class EnhancerDataset(Dataset):
    def __init__(self, base_path: str, default_samplerate: int = 48000):
        self._base_path = base_path
        self._examples = []
        self._sample_rate = default_samplerate
        files_tmp = [join(base_path, f) for f in listdir(base_path) if
                     isfile(join(base_path, f)) and f.endswith('.wav')]
        self._examples = files_tmp

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        # read initial sample rate
        w = wave.open(self._examples[item])
        sample_rate = w.getframerate()
        w.close()
        # read audio and resample to 48000
        audio, sr = librosa.load(self._examples[item], sr=self._sample_rate)
        # alter original signal
        x = alter(copy.deepcopy(audio), prob=0.5, real_sr=sr)
        return {
            'x': x,
            'y': audio,
            'sample_rate': sample_rate
        }


def collate_fn(batch):
    pass


if __name__ == '__main__':
    dataset = EnhancerDataset('data/processed/train')
    from ipdb import set_trace

    set_trace()
