import random
import torch
from torch.utils.data.dataset import Dataset
import sys
import numpy as np

sys.path.append('')
import json


class PhonemizerDataset(Dataset):
    def __init__(self, filename: str):
        self._examples = json.load(open(filename))

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, index):
        return self._examples[index]


class PhonemizerEncodings():
    def __init__(self, filename: str = None):
        self._grapheme2int = {}
        self._phon2int = {}
        if filename is not None:
            self.load(filename)

    def save(self, filename: str):
        obj = {
            'grapheme2int': self._grapheme2int,
            'phon2int': self._phon2int
        }
        json.dump(obj, open(filename, 'w'))

    def load(self, filename: str):
        obj = json.load(open(filename))
        self._grapheme2int = obj['grapheme2int']
        self._phon2int = obj['phon2int']

    def compute(self, dataset: PhonemizerDataset):
        self._phon2int = {'PAD': 0}
        self._grapheme2int = {'PAD': 0}
        for example in dataset:
            text = example['orig_text']
            phones = example['phones']
            for g in text:
                g = g.lower()
                if g not in self._grapheme2int:
                    self._grapheme2int[g] = len(self._grapheme2int)
            for p in phones:
                if p not in self._phon2int:
                    self._phon2int[p] = len(self._phon2int)

    @property
    def phonemes(self):
        return self._phon2int

    @property
    def graphemes(self):
        return self._grapheme2int


class PhonemizerCollate:
    def __init__(self, encodings: PhonemizerEncodings):
        self._encodings = encodings

    def collate_fn(self, batch):
        max_char = max([len(example['orig_text']) for example in batch])
        max_phon = max([len(example['phones']) for example in batch])
        x_char = np.zeros((len(batch), max_char))
        x_case = np.zeros((len(batch), max_char))
        y_phon = np.zeros((len(batch), max_phon))
        y_new_word = np.zeros((len(batch), max_phon))
        x_words = []

        for ii in range(len(batch)):
            example = batch[ii]
            text = example['orig_text']
            if 'hybrid' in example:
                phones = example['hybrid']
            else:
                phones = example['phones']
            phon2word = example['phon2word']
            # x_words.append(example['words'])
            offset = 0
            x_words.append([])
            for w in example['words']:
                x_words[-1].append({'word': w, 'start': offset, 'stop': offset + len(w)})
                offset += len(w)
            for jj in range(len(text)):
                g = text[jj]
                g_low = g.lower()
                if g_low != g:
                    x_case[ii, jj] = 1
                if g_low in self._encodings._grapheme2int:
                    x_char[ii, jj] = self._encodings._grapheme2int[g_low]
            for jj in range(len(phones)):
                p = phones[jj]
                current_p2w = phon2word[jj]
                next_p2w = current_p2w + 1
                if jj < len(phones) - 1:
                    next_p2w = phon2word[jj + 1]
                if current_p2w != next_p2w:
                    y_new_word[ii, jj] = next_p2w - current_p2w + 1
                else:
                    y_new_word[ii, jj] = 1
                if p in self._encodings._phon2int:
                    y_phon[ii, jj] = self._encodings._phon2int[p]

        return {
            'x_char': torch.tensor(x_char, dtype=torch.long),
            'x_case': torch.tensor(x_case, dtype=torch.long),
            'y_phon': torch.tensor(y_phon, dtype=torch.long),
            'y_new_word': torch.tensor(y_new_word, dtype=torch.long),
            'x_words': x_words
        }
