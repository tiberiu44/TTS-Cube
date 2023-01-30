import os.path

import torch
import torch.nn as nn
import numpy as np
import sys
import json
from os import listdir
from os.path import isfile, join, exists

import tqdm

sys.path.append('')

from torch.utils.data.dataset import Dataset


class TextcoderDataset(Dataset):
    def __init__(self, base_path: str):
        self._base_path = base_path
        self._examples = []
        train_files_tmp = [join(base_path, f) for f in listdir(base_path) if isfile(join(base_path, f))]

        for file in train_files_tmp:
            if file[-4:] == '.mgc':
                bpath = file[:-4]
                # check all files exist
                json_file = '{0}.json'.format(bpath)
                pitch_file = '{0}.pitch'.format(bpath)
                if os.path.exists(json_file) and os.path.exists(pitch_file):
                    self._examples.append(json.load(open(json_file)))

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        description = self._examples[item]
        base_fn = '{0}/{1}'.format(self._base_path, description['id'])
        mgc = np.load('{0}.mgc'.format(base_fn))
        pitch = np.load('{0}.pitch'.format(base_fn))
        return {
            'meta': description,
            'mgc': mgc,
            'pitch': pitch
        }


class TextcoderEncodings:
    def __init__(self, filename: str = None):
        self.speaker2int = {}
        self.phon2int = {}
        self.max_duration = 0
        self.max_pitch = 0

    def compute(self, dataset: Text2MelDataset):
        for example in tqdm.tqdm(dataset, ncols=80, desc='Computing encodings'):
            speaker = example['meta']['speaker']
            if speaker not in self.speaker2int:
                self.speaker2int[speaker] = len(self.speaker2int)
            for phone in example['meta']['phones']:
                if phone not in self.phon2int:
                    self.phon2int[phone] = len(self.phon2int)
            m_pitch = np.max(example['pitch'])
            if m_pitch > self.max_pitch:
                self.max_pitch = m_pitch
            durs = np.zeros((len(example['meta']['phones'])), dtype=np.long)
            for item in example['meta']['frame2phon']:
                durs[item] += 1
            m_dur = np.max(durs)
            if m_dur > self.max_duration:
                self.max_duration = m_dur

    def load(self, filename: str):
        pass

    def save(self, filename: str):
        pass


class TextcoderCollate:
    def __init__(self, encodings):
        self._encodings = encodings

    def collate_fn(self, batch):
        pass
