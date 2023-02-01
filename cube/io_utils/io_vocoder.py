import random
import torch

import librosa
from torch.utils.data.dataset import Dataset
import sys
from os import listdir
from os.path import isfile, join
import os
import numpy as np

sys.path.append('')
from cube.io_utils.vocoder import MelVocoder


class VocoderDataset(Dataset):
    def __init__(self,
                 path: str, target_sample_rate: int = 24000,
                 lowres_sample_rate: int = 2400,
                 max_segment_size=-1,
                 random_start=True,
                 hop_size=240):
        self._examples = []
        self._sample_rate = target_sample_rate
        self._sample_rate_low = lowres_sample_rate
        self._max_segment_size = max_segment_size
        self._mel_vocoder = MelVocoder()
        self._hop_size = hop_size
        self._random_start = random_start
        train_files_tmp = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

        for file in train_files_tmp:
            if file[-4:] == '.wav':
                w_size = os.stat(file).st_size
                if w_size > 4096 and w_size > max_segment_size * 2:
                    self._examples.append(file)
        if not os.path.exists('data/cache'):
            os.makedirs('data/cache')

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):

        filename = self._examples[item]
        cache_filename = 'data/cache/{0}'.format(filename.replace('/', '_').replace('\\', '_'))
        if os.path.exists('{0}.mgc.npy'.format(cache_filename)):
            mel = np.load('{0}.mgc.npy'.format(cache_filename))
            wav = np.load('{0}.audio.npy'.format(cache_filename))
            wav_low = np.load('{0}.audio_low.npy'.format(cache_filename))
        else:
            wav, sr = librosa.load(filename, sr=self._sample_rate)
            wav_low, sr = librosa.load(filename, sr=self._sample_rate_low)
            wav = (wav / np.max(np.abs(wav))) * 0.98
            wav_low = (wav_low / np.max(np.abs(wav_low))) * 0.98
            mel = self._mel_vocoder.melspectrogram(wav,
                                                   sample_rate=self._sample_rate,
                                                   num_mels=80,
                                                   hop_size=self._hop_size,
                                                   use_preemphasis=False)
            np.save('{0}.mgc'.format(cache_filename), mel)
            np.save('{0}.audio'.format(cache_filename), wav)
            np.save('{0}.audio_low'.format(cache_filename), wav_low)
        if self._max_segment_size == -1 or len(wav) < self._max_segment_size or not self._random_start:
            if not self._random_start and self._max_segment_size != -1 and len(wav > self._max_segment_size):
                return wav[:self._max_segment_size], \
                       wav_low[:self._max_segment_size // (self._sample_rate // self._sample_rate_low)], \
                       mel[:self._max_segment_size // self._hop_size + 1]
            else:
                return wav, \
                       wav_low, \
                       mel
        else:
            start = random.randint(0, len(wav) - self._max_segment_size - 1)
            hs = self._sample_rate // self._sample_rate_low
            start = start // self._hop_size * self._hop_size  # multiple of hop size
            stop = start + self._max_segment_size
            start_low = start // hs
            stop_low = start_low + self._max_segment_size // hs
            return wav[start:stop], \
                   wav_low[start_low:stop_low], \
                   mel[start // self._hop_size:stop // self._hop_size + 1]


class VocoderCollate:
    def __init__(self, x_zero=0, mel_zero=-5):
        self._x_zero = x_zero
        self._mel_zero = mel_zero

    def collate_fn(self, examples):
        max_audio_size = max([x[0].shape[0] for x in examples])
        max_audio_low_size = max([x[1].shape[0] for x in examples])
        max_mel_size = max([x[2].shape[0] for x in examples])
        mel = np.ones((len(examples), max_mel_size, examples[0][2].shape[1]), dtype=np.float) * self._mel_zero
        x = np.ones((len(examples), max_audio_size)) * self._x_zero
        x_low = np.ones((len(examples), max_audio_low_size)) * self._x_zero
        for ii in range(len(examples)):
            cx = examples[ii][0]
            cxl = examples[ii][1]
            cmel = examples[ii][2]
            mel[ii, :cmel.shape[0], :] = cmel
            x[ii, :cx.shape[0]] = cx
            x_low[ii, :cxl.shape[0]] = cxl

        x = torch.tensor(x, dtype=torch.float)
        mel = torch.tensor(mel, dtype=torch.float)
        x_low = torch.tensor(x_low, dtype=torch.float)
        return {
            'x': x,
            'x_low': x_low,
            'mel': mel
        }
