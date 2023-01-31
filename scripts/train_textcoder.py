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
from cube.networks.textcoder import CubenetTextcoder
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from cube.io_utils.io_textcoder import TextcoderCollate, TextcoderEncodings, TextcoderDataset
from cube.io_utils.runtime import synthesize_devset


class PrintAndSaveCallback(pl.callbacks.Callback):
    def __init__(self, store_prefix, generate_epoch):
        super().__init__()
        self.store_prefix = store_prefix
        self._best_loss = 99999
        self._generate_epoch = generate_epoch

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss = pl_module._val_loss_mel
        sys.stdout.write('\n\n\tVal loss mel: {0}\n\tVal loss pitch: {1}\n\tVal loss dur: {2}\n'.
                         format(pl_module._val_loss_mel, pl_module._val_loss_pitch, pl_module._val_loss_durs))
        sys.stdout.flush()
        if val_loss < self._best_loss:
            self._best_loss = val_loss
            fname = "{0}.best".format(self.store_prefix)
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
        opt = pl_module.optimizers()
        torch.save(opt.state_dict(), fname)
        if epoch % self._generate_epoch == 0:
            sys.stdout.write('\tGenerating validation set\n')
            sys.stdout.flush()
            os.makedirs('generated_files/free/', exist_ok=True)
            synthesize_devset(self.store_prefix,
                              'data/models/vocoder/neb-noft/g_00600000',
                              output_path='generated_files/free/',
                              forced_synthesis=False,
                              limit=10)


def _train(params):
    config = {
        'sample_rate': params.sample_rate,
        'pframes': params.pframes,
        'hop_size': params.hop_size
    }
    conf_file = '{0}.yaml'.format(params.output_base)
    yaml.dump(config, open(conf_file, 'w'))
    sys.stdout.write('=================Config=================\n')
    sys.stdout.write(open(conf_file).read())
    sys.stdout.write('========================================\n\n')
    trainset = TextcoderDataset(params.train_folder)
    devset = TextcoderDataset(params.dev_folder)
    sys.stdout.write('==================Data==================\n')
    sys.stdout.write('Training files: {0}\n'.format(len(trainset)))
    sys.stdout.write('Validation files: {0}\n'.format(len(devset)))
    sys.stdout.write('========================================\n\n')
    sys.stdout.write('================Training================\n')
    encodings = TextcoderEncodings()
    if params.resume:
        encodings.load('{0}.encodings'.format(params.output_base))
    else:
        encodings.compute(trainset)
        encodings.save('{0}.encodings'.format(params.output_base))
    collate = TextcoderCollate(encodings)
    sys.stdout.write('Number of speakers: {0}\n'.format(len(encodings.speaker2int)))
    sys.stdout.write('Number of phones: {0}\n'.format(len(encodings.phon2int)))
    sys.stdout.write('Maximum F0: {0}\n'.format(encodings.max_pitch))
    sys.stdout.write('Maximum duration: {0}\n'.format(encodings.max_duration))
    trainloader = DataLoader(trainset,
                             batch_size=params.batch_size,
                             num_workers=params.num_workers,
                             collate_fn=collate.collate_fn)
    devloader = DataLoader(devset,
                           batch_size=params.batch_size,
                           num_workers=params.num_workers,
                           collate_fn=collate.collate_fn)

    model = CubenetTextcoder(encodings)

    if params.resume:
        sys.stdout.write('Resuming from previous checkpoint\n')
        sys.stdout.flush()
        model.load('{0}.last'.format(params.output_base))

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
                        help='Where to store the model (default=data/vocoder)')
    parser.add_argument('--batch-size', dest='batch_size', default=16,
                        type=int, help='Batch size (default=16)')
    parser.add_argument('--num-workers', dest='num_workers', default=4,
                        type=int, help='Batch size (default=4)')
    parser.add_argument('--maximum-segment-size', dest='maximum_segment_size', type=int,
                        default=24000, help='Maximum audio segment size - will be randomly selected (default=24000)')
    parser.add_argument('--accelerator', dest='accelerator', default='cpu',
                        help='What accelerator to use (default=cpu) - check pytorch lightning for possible values')
    parser.add_argument('--devices', dest='devices', default=1, type=int,
                        help='How many devices to use (default=1)')
    parser.add_argument('--train-folder', dest='train_folder', default='data/processed/train',
                        help='Location of training files (default=data/processed/train)')
    parser.add_argument('--dev-folder', dest='dev_folder', default='data/processed/dev',
                        help='Location of training files (default=data/processed/dev)')
    parser.add_argument('--sample-rate', dest='sample_rate', type=int, default=24000,
                        help='Number of parallel samples (default=24000)')
    parser.add_argument('--hop-size', dest='hop_size', type=int, default=240,
                        help='Hop-size for mel (default=240)')
    parser.add_argument('--lr', dest='lr', default=2e-4, type=float,
                        help='Learning rate (default=2e-4)')
    parser.add_argument('--pframes', dest='pframes', type=int, default=3,
                        help='How many frames to generate at the same time (default=3)')
    parser.add_argument('--epoch-generation', dest='epoch_generation', type=int, default=10,
                        help='End-to-end generation of validation set at every n-th epoch (default=10). '
                             'Files are stored in generated_files/free')

    parser.add_argument('--resume', dest='resume', action='store_true')

    args = parser.parse_args()

    _train(args)
