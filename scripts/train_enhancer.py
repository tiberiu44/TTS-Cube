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
from cube.networks.enhancer import Cubedall
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from cube.io_utils.io_enhancer import EnhancerDataset, collate_fn
from cube.io_utils.runtime import cubegan_synthesize_dataset


class PrintAndSaveCallback(pl.callbacks.Callback):
    def __init__(self, store_prefix, generate_epoch):
        super().__init__()
        self.store_prefix = store_prefix
        self._best_loss = 99999
        self._generate_epoch = generate_epoch

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss = pl_module._val_loss
        sys.stdout.write('\n\n\tVal loss: {0}\n'.
                         format(pl_module._val_loss))
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
        opts = pl_module.optimizers()

        if isinstance(opts, list):
            opt_dict = {str(ii): opt.state_dict() for ii, opt in enumerate(opts)}
        else:
            opt_dict = {'0': opts.state_dict()}
        opt_dict['global_step'] = pl_module._global_step
        torch.save(opt_dict, fname)
        # torch.save(opt.state_dict(), fname)
        if epoch % self._generate_epoch == 0:
            sys.stdout.write('\tGenerating validation set\n')
            sys.stdout.flush()
            os.makedirs('generated_files/free/', exist_ok=True)
            cubegan_synthesize_dataset(pl_module,
                                       output_path='generated_files/free/',
                                       devset_path='data/processed/dev/',
                                       limit=10,
                                       conditioning=pl_module._conditioning)


def _train(params):
    trainset = EnhancerDataset(params.train_folder)
    devset = EnhancerDataset(params.dev_folder)
    sys.stdout.write('==================Data==================\n')
    sys.stdout.write('Training files: {0}\n'.format(len(trainset)))
    sys.stdout.write('Validation files: {0}\n'.format(len(devset)))
    sys.stdout.write('========================================\n\n')
    sys.stdout.write('================Training================\n')

    trainloader = DataLoader(trainset,
                             batch_size=params.batch_size,
                             num_workers=params.num_workers,
                             collate_fn=collate_fn)
    devloader = DataLoader(devset,
                           batch_size=params.batch_size,
                           num_workers=params.num_workers,
                           collate_fn=collate_fn)

    model = Cubedall(lr=params.lr)

    if params.resume:
        sys.stdout.write('Resuming from previous checkpoint\n')
        sys.stdout.flush()
        model.load('{0}.last'.format(params.output_base))
        opts_state = torch.load('{0}.opt.last'.format(params.output_base))
        # opts = model.optimizers()
        model._loaded_optimizer_state = opts_state
        model._global_step = opts_state['global_step']

    trainer = pl.Trainer(
        accelerator=params.accelerator,
        devices=params.devices,
        max_epochs=-1,
        callbacks=[PrintAndSaveCallback(params.output_base, params.epoch_generation)]
    )

    trainer.fit(model, trainloader, devloader)


if __name__ == '__main__':
    parser = ArgumentParser(description='NLP-Cube Trainer Helper')
    parser.add_argument('--output-base', action='store', dest='output_base',
                        default='data/cubedall',
                        help='Where to store the model (default=data/cubedall)')
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
    parser.add_argument('--lr', dest='lr', default=2e-4, type=float,
                        help='Learning rate (default=2e-4)')
    parser.add_argument('--epoch-generation', dest='epoch_generation', type=int, default=10,
                        help='End-to-end generation of validation set at every n-th epoch (default=10). '
                             'Files are stored in generated_files/free')

    parser.add_argument('--resume', dest='resume', action='store_true')

    args = parser.parse_args()

    _train(args)
