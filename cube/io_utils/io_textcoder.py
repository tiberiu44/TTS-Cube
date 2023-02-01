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

    def compute(self, dataset: TextcoderDataset):
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
        input_obj = json.load(open(filename))
        self.speaker2int = input_obj['speaker2int']
        self.phon2int = input_obj['phon2int']
        self.max_pitch = input_obj['max_pitch']
        self.max_duration = input_obj['max_duration']

    def save(self, filename: str):
        output_obj = {
            'speaker2int': self.speaker2int,
            'phon2int': self.phon2int,
            'max_duration': int(self.max_duration),
            'max_pitch': int(self.max_pitch)
        }
        json.dump(output_obj, open(filename, 'w'))


class TextcoderCollate:
    def __init__(self, encodings: TextcoderEncodings):
        self._encodings = encodings
        self._ignore_index = int(max(encodings.max_pitch, encodings.max_duration) + 1)

    def collate_fn(self, batch):
        max_char = max([len(example['meta']['phones']) for example in batch])
        max_mel = max([example['mgc'].shape[0] for example in batch])
        x_char = np.zeros((len(batch), max_char))
        y_mgc = np.ones((len(batch), max_mel, 80)) * -5
        x_speaker = np.zeros((len(batch), 1))
        y_dur = np.zeros((len(batch), max_char))
        y_pitch = np.ones((len(batch), max_mel)) * self._ignore_index
        y_frame2phone = []
        for ii in range(len(batch)):
            example = batch[ii]
            y_mgc[ii, :example['mgc'].shape[0], :] = example['mgc']
            x_speaker[ii] = self._encodings.speaker2int[example['meta']['speaker']] + 1
            for jj in range(len(example['meta']['phones'])):
                phoneme = example['meta']['phones'][jj]
                if phoneme in self._encodings.phon2int:
                    x_char[ii, jj] = self._encodings.phon2int[phoneme] + 1
            y_frame2phone.append(example['meta']['frame2phon'])
            for phone_idx in y_frame2phone[-1]:
                y_dur[ii, phone_idx] += 1
            for jj in range(max_char - len(example['meta']['phones'])):
                y_dur[ii, len(example['meta']['phones']) + jj] = self._ignore_index
            y_pitch[ii, :example['pitch'].shape[0]] = example['pitch']

        return {
            'x_char': torch.tensor(x_char, dtype=torch.long),
            'x_speaker': torch.tensor(x_speaker, dtype=torch.long),
            'y_mgc': torch.tensor(y_mgc, dtype=torch.float),
            'y_frame2phone': y_frame2phone,
            'y_pitch': torch.tensor(y_pitch, dtype=torch.long),
            'y_dur': torch.tensor(y_dur, dtype=torch.long)

        }
