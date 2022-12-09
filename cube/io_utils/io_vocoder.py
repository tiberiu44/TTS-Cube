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
    def __init__(self, path: str, target_sample_rate: int = 22050, max_segment_size=-1, random_start=True,
                 use_cache=True):
        self._examples = []
        self._sample_rate = target_sample_rate
        self._max_segment_size = max_segment_size
        self._mel_vocoder = MelVocoder()
        self._random_start = random_start
        train_files_tmp = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

        for file in train_files_tmp:
            if file[-4:] == '.wav':
                w_size = os.stat(file).st_size
                if w_size > 4096 and w_size > max_segment_size * 2:
                    self._examples.append(file)
        self._use_cache = use_cache
        if use_cache and not os.path.exists('data/cache'):
            os.makedirs('data/cache')

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):

        filename = self._examples[item]
        cache_filename = 'data/cache/{0}'.format(filename.replace('/', '_').replace('\\', '_'))
        if self._use_cache:
            if os.path.exists('{0}.mgc.npy'.format(cache_filename)):
                mel = np.load('{0}.mgc.npy'.format(cache_filename))
                wav = np.load('{0}.audio.npy'.format(cache_filename))
            else:
                wav, sr = librosa.load(filename, sr=self._sample_rate)
                wav = wav / np.max(np.abs(wav))
                mel = self._mel_vocoder.melspectrogram(wav,
                                                       sample_rate=self._sample_rate,
                                                       num_mels=80,
                                                       use_preemphasis=False)
                np.save('{0}.mgc'.format(cache_filename), mel)
                np.save('{0}.audio'.format(cache_filename), wav)
            if self._max_segment_size == -1 or len(wav) < self._max_segment_size or not self._random_start:
                return (wav, mel)
            else:
                start = random.randint(0, len(wav) - self._max_segment_size - 1)
                start = start // 256 * 256  # multiple of hop size
                stop = start + self._max_segment_size
                return (wav[start:stop], mel[start // 256:stop // 256 + 1])
        else:
            wav, sr = librosa.load(filename, sr=self._sample_rate)
            wav = wav / np.max(np.abs(wav))
            if self._max_segment_size != -1 and len(wav) > self._max_segment_size:
                if self._random_start:
                    start = random.randint(0, len(wav) - self._max_segment_size - 1)
                else:
                    start = 0
                x = wav[start:start + self._max_segment_size]
            else:
                x = wav

            mel = self._mel_vocoder.melspectrogram(x, sample_rate=self._sample_rate, num_mels=80, use_preemphasis=False)
            return (x, mel)


class VocoderCollate:
    def __init__(self):
        pass

    def collate_fn(self, examples):
        max_audio_size = max([x[0].shape[0] for x in examples])
        max_mel_size = max([x[1].shape[0] for x in examples])
        mel = np.ones((len(examples), max_mel_size, examples[0][1].shape[1]), dtype=np.float) * -5
        x = np.zeros((len(examples), max_audio_size))
        for ii in range(len(examples)):
            cx = examples[ii][0]
            cmel = examples[ii][1]
            mel[ii, :cmel.shape[0], :] = cmel
            x[ii, :cx.shape[0]] = cx

        x = torch.tensor(x, dtype=torch.float)
        mel = torch.tensor(mel, dtype=torch.float)
        x = x / torch.max(torch.abs(x))  # normalize
        return {
            'x': x,
            'mel': mel
        }
