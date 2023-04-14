import copy
import random

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
        try:
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
        except:
            return {
                'x': np.zeros((48000)),
                'y': np.zeros((48000)),
                'sample_rate': 48000
            }


def collate_fn(batch, max_segment_size=24000):
    x = np.zeros((len(batch), max_segment_size))
    y = np.zeros((len(batch), max_segment_size))
    sr = np.zeros((len(batch), 5))
    for index in range(len(batch)):
        example = batch[index]
        ssr = example['sample_rate']
        if ssr == 8000:
            sr[index, 0] = 1
        elif ssr == 16000:
            sr[index, 1] = 1
        elif ssr == 22050:
            sr[index, 2] = 1
        elif ssr == 24000:
            sr[index, 3] = 1
        else:
            sr[index, 4] = 1
        sx = example['x']
        sy = example['y']
        if sy.shape[0] <= max_segment_size:
            x[index, :sx.shape[0]] = sx
            y[index, :sy.shape[0]] = sy
        else:
            start = random.randint(0, sy.shape[0] - max_segment_size - 1)
            x[index, :] = sx[start:start + max_segment_size]
            y[index, :] = sy[start:start + max_segment_size]

    return {
        'x': torch.tensor(x, dtype=torch.float),
        'y': torch.tensor(y, dtype=torch.float),
        'sr': torch.tensor(sr, dtype=torch.float)
    }


if __name__ == '__main__':
    dataset = EnhancerDataset('data/processed/train')
    from ipdb import set_trace

    set_trace()
