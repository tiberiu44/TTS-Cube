import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append('')

from torch.utils.data.dataset import Dataset


class Text2MelDataset(Dataset):
    def __init__(self, base_path: str):
        self._base_path = base_path
        self._examples = []

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
