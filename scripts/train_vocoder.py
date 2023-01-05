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

import sys
import json
import yaml
import pytorch_lightning as pl

sys.path.append('')
from cube.networks.vocoder import CubenetVocoder
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from argparse import BooleanOptionalAction
from cube.io_utils.io_vocoder import VocoderDataset, VocoderCollate


class PrintAndSaveCallback(pl.callbacks.Callback):
    def __init__(self, store_prefix):
        super().__init__()
        self.store_prefix = store_prefix
        self._best_loss_lr = 99999
        self._best_loss_hr = 99999

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss_lr = pl_module._val_loss_lr
        val_loss_hr = pl_module._val_loss_hr
        sys.stdout.write('\n\n\tVal loss lr: {0}\n\tVal loss hr: {1}\n'.format(val_loss_lr, val_loss_hr))
        sys.stdout.flush()
        if val_loss_lr < self._best_loss_lr:
            self._best_loss_lr = val_loss_lr
            fname = "{0}.lr.best".format(self.store_prefix)
            sys.stdout.write('\tStoring {0}\n'.format(fname))
            sys.stdout.flush()
            pl_module._wavernn_lr.save(fname)
        if val_loss_hr < self._best_loss_hr:
            self._best_loss_hr = val_loss_hr
            fname = "{0}.hr.best".format(self.store_prefix)
            sys.stdout.write('\tStoring {0}\n'.format(fname))
            sys.stdout.flush()
            pl_module._wavernn_hr.save(fname)

        fname = "{0}.last".format(self.store_prefix)
        sys.stdout.write('\tStoring {0}\n'.format(fname))
        sys.stdout.flush()
        pl_module.save(fname)


def _train(params):
    config = {
        'num_layers_lr': params.num_layers_lr,
        'layer_size_lr': params.layer_size_lr,
        'num_layers_hr': params.num_layers_hr,
        'layer_size_hr': params.layer_size_hr,
        'upsample': params.upsample,
        'sample_rate': params.sample_rate,
        'output': params.output,
        'sample_rate_low': params.sample_rate_low,
        'hop_size': params.hop_size
    }
    conf_file = '{0}.yaml'.format(params.output_base)
    yaml.dump(config, open(conf_file, 'w'))
    sys.stdout.write('=================Config=================\n')
    sys.stdout.write(open(conf_file).read())
    sys.stdout.write('========================================\n\n')
    trainset = VocoderDataset(params.train_folder, target_sample_rate=params.sample_rate,
                              lowres_sample_rate=params.sample_rate_low,
                              hop_size=params.hop_size,
                              max_segment_size=params.maximum_segment_size, random_start=True)
    devset = VocoderDataset(params.dev_folder, target_sample_rate=params.sample_rate,
                            lowres_sample_rate=params.sample_rate_low,
                            hop_size=params.hop_size,
                            max_segment_size=params.maximum_segment_size, random_start=False)
    sys.stdout.write('==================Data==================\n')
    sys.stdout.write('Training files: {0}\n'.format(len(trainset)))
    sys.stdout.write('Validation files: {0}\n'.format(len(devset)))
    sys.stdout.write('========================================\n\n')
    sys.stdout.write('================Training================\n')
    collate = VocoderCollate(x_zero=0)
    trainloader = DataLoader(trainset,
                             batch_size=params.batch_size,
                             num_workers=params.num_workers,
                             collate_fn=collate.collate_fn)
    devloader = DataLoader(devset,
                           batch_size=params.batch_size,
                           num_workers=params.num_workers,
                           collate_fn=collate.collate_fn)
    model = CubenetVocoder(num_layers_hr=params.num_layers_hr,
                           layer_size_hr=params.layer_size_hr,
                           num_layers_lr=params.num_layers_lr,
                           layer_size_lr=params.layer_size_lr,
                           upsample=params.upsample,
                           upsample_low=params.sample_rate // params.sample_rate_low,
                           learning_rate=params.lr,
                           output=params.output)

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
                        default='data/vocoder',
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
    parser.add_argument('--sample-rate-low', dest='sample_rate_low', type=int, default=2400,
                        help='Number of parallel samples (default=2400)')
    parser.add_argument('--layer-size-hr', dest='layer_size_hr', default=512, type=int,
                        help='LSTM layer size (default=512)')
    parser.add_argument('--num-layers-hr', dest='num_layers_hr', default=1, type=int,
                        help='Number of LSTM layers (default=1)')
    parser.add_argument('--layer-size-lr', dest='layer_size_lr', default=512, type=int,
                        help='LSTM layer size (default=512)')
    parser.add_argument('--num-layers-lr', dest='num_layers_lr', default=1, type=int,
                        help='Number of LSTM layers (default=1)')
    parser.add_argument('--hop-size', dest='hop_size', type=int, default=240,
                        help='Hop-size for mel (default=240)')
    parser.add_argument('--upsample', dest='upsample', default=240, type=int,
                        help='Upsample layers (default=10)')
    parser.add_argument('--lr', dest='lr', default=1e-4, type=float,
                        help='Learning rate (default=1e-4)')
    parser.add_argument('--output', dest='output', default='mol',
                        help='Output type (mol|gm|mulaw|beta) (default=mol)')

    parser.add_argument('--resume', dest='resume', action='store_true')

    args = parser.parse_args()

    _train(args)
