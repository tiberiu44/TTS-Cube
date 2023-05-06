import copy
import random
import time

import torch
import numpy as np
from os import listdir
from os.path import isfile, join
import torchaudio
import torchaudio.transforms as T

from torch.utils.data.dataset import Dataset
import sys

sys.path.append('')
from cube.io_utils.audio import alter, pitch_shift


class EnhancerDataset(Dataset):
    def __init__(self, base_path: str, default_samplerate: int = 48000):
        self._base_path = base_path
        self._examples = []
        self._sample_rate = default_samplerate
        files_tmp_clean = [join(join(base_path, 'clean'), f) for f in listdir(join(base_path, 'clean')) if
                           isfile(join(join(base_path, 'clean'), f)) and f.endswith('.wav')]
        for file in files_tmp_clean:
            self._examples.append((file, 1))
        files_tmp_general = [join(join(base_path, 'general'), f) for f in listdir(join(base_path, 'general')) if
                             isfile(join(join(base_path, 'general'), f)) and f.endswith('.wav')]
        for file in files_tmp_general:
            self._examples.append((file, 0))

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        try:
            audio, sample_rate = torchaudio.load(self._examples[item][0])
            if torch.max(audio) > 500:
                audio = audio / 32767
            is_clean = self._examples[item][1] == 1
            audio = audio[0, :].unsqueeze(0)
            res = T.Resample(sample_rate, self._sample_rate, dtype=audio.dtype)
            audio = res(audio)
            if not is_clean:
                return {
                    'x': audio.squeeze(0).numpy(),
                    'y': audio.squeeze(0).numpy(),
                    'sample_rate': sample_rate,
                    'denoise': False
                }
            p = random.random()
            if p < 0.3:
                audio = pitch_shift(audio, sample_rate)
            x = alter(copy.deepcopy(audio), prob=0.5, real_sr=sample_rate)
            return {
                'x': x.squeeze(0).numpy(),
                'y': audio.squeeze(0).numpy(),
                'sample_rate': sample_rate,
                'denoise': True
            }
        except:
            print("err")
            return {
                'x': np.zeros((48000)),
                'y': np.zeros((48000)),
                'sample_rate': 48000,
                'denoise': True
            }


def collate_fn(batch, max_segment_size=24000):
    x = np.zeros((len(batch), max_segment_size))
    y = np.zeros((len(batch), max_segment_size))
    sr = np.zeros((len(batch), 5))
    denoise = np.zeros((len(batch), 1))
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
        sx = example['x'].squeeze()
        sy = example['y'].squeeze()
        if example['denoise']:
            denoise[index, 0] = 1
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
        'sr': torch.tensor(sr, dtype=torch.float),
        'denoise': torch.tensor(denoise, dtype=torch.float)
    }


if __name__ == '__main__':
    dataset = EnhancerDataset('data/processed/train')
    avg_time = 0
    for zz in range(5):
        start = time.time()
        for ii in range(10):
            print(ii)
            tmp = dataset[ii]
        stop = time.time()
        avg_time += stop - start
    print(avg_time / 5)
    from ipdb import set_trace

    set_trace()
