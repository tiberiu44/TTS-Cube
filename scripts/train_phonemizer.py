#
# Author: Tiberiu Boros
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import sys
import json
import yaml
import pytorch_lightning as pl
import torch

sys.path.append('')
from cube.networks.phonemizer import CubenetPhonemizer
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from cube.io_utils.io_phonemizer import PhonemizerEncodings, PhonemizerDataset, PhonemizerCollate


class PrintAndSaveCallback(pl.callbacks.Callback):
    def __init__(self, store_prefix):
        super().__init__()
        self.store_prefix = store_prefix
        self._best_loss = 99999
        self._best_sacc = 0
        self._best_pacc = 0

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss = pl_module._val_loss
        val_pacc = pl_module._val_pacc
        val_sacc = pl_module._val_sacc
        sys.stdout.write('\n\n\tVal loss: {0}\n\tVal PACC: {1}\n\tVal SACC: {2}\n'.
                         format(val_loss, val_pacc, val_sacc))
        sys.stdout.flush()
        if val_loss < self._best_loss:
            self._best_loss = val_loss
            fname = "{0}.best".format(self.store_prefix)
            sys.stdout.write('\tStoring {0}\n'.format(fname))
            sys.stdout.flush()
            pl_module.save(fname)

        if val_pacc > self._best_pacc:
            self._best_pacc = val_pacc
            fname = "{0}.pacc.best".format(self.store_prefix)
            sys.stdout.write('\tStoring {0}\n'.format(fname))
            sys.stdout.flush()
            pl_module.save(fname)

        if val_sacc > self._best_sacc:
            self._best_sacc = val_sacc
            fname = "{0}.sacc.best".format(self.store_prefix)
            sys.stdout.write('\tStoring {0}\n'.format(fname))
            sys.stdout.flush()
            pl_module.save(fname)

        fname = "{0}.last".format(self.store_prefix)
        sys.stdout.write('\tStoring {0}\n'.format(fname))
        sys.stdout.flush()
        pl_module.save(fname)
        fname = "{0}.opt.last".format(self.store_prefix)
        sys.stdout.write('\tStoring {0}\n'.format(fname))
        sys.stdout.flush()


def _train(params):
    trainset = PhonemizerDataset(params.train_file)
    devset = PhonemizerDataset(params.dev_file)
    sys.stdout.write('==================Data==================\n')
    sys.stdout.write('Training examples: {0}\n'.format(len(trainset)))
    sys.stdout.write('Validation examples: {0}\n'.format(len(devset)))
    sys.stdout.write('========================================\n\n')
    sys.stdout.write('================Training================\n')
    encodings = PhonemizerEncodings()
    encodings.compute(trainset)
    encodings.save('{0}.encodings'.format(params.output_base))
    collate = PhonemizerCollate(encodings)
    sys.stdout.write('Number of graphemes: {0}\n'.format(len(encodings.graphemes)))
    sys.stdout.write('Number of phones: {0}\n'.format(len(encodings.phonemes)))
    trainloader = DataLoader(trainset,
                             batch_size=params.batch_size,
                             num_workers=params.num_workers,
                             collate_fn=collate.collate_fn)
    devloader = DataLoader(devset,
                           batch_size=params.batch_size,
                           num_workers=params.num_workers,
                           collate_fn=collate.collate_fn)

    model = CubenetPhonemizer(encodings, lr=params.lr)

    trainer = pl.Trainer(
        accelerator=params.accelerator,
        devices=params.devices,
        max_epochs=-1,
        callbacks=[PrintAndSaveCallback(params.output_base)]
    )

    trainer.fit(model, trainloader, devloader)


if __name__ == '__main__':
    parser = ArgumentParser(description='NLP-Cube Trainer Helper')
    parser.add_argument('--output-base', action='store', dest='output_base',
                        default='data/textcoder',
                        help='Where to store the model (default=data/phonemizer)')
    parser.add_argument('--batch-size', dest='batch_size', default=16,
                        type=int, help='Batch size (default=16)')
    parser.add_argument('--num-workers', dest='num_workers', default=4,
                        type=int, help='Batch size (default=4)')
    parser.add_argument('--accelerator', dest='accelerator', default='cpu',
                        help='What accelerator to use (default=cpu) - check pytorch lightning for possible values')
    parser.add_argument('--devices', dest='devices', default=1, type=int,
                        help='How many devices to use (default=1)')
    parser.add_argument('--train-file', dest='train_file', default='data/blizzard-g2p.train',
                        help='Location of training file (default=data/blizzard-g2p.train)')
    parser.add_argument('--dev-file', dest='dev_file', default='data/blizzard-g2p.dev',
                        help='Location of validation file (default=data/blizzard-g2p.dev)')
    parser.add_argument('--lr', dest='lr', default=2e-4, type=float,
                        help='Learning rate (default=2e-4)')

    args = parser.parse_args()

    _train(args)
