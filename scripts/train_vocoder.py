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
from cube.io_utils.io_vocoder import VocoderDataset, VocoderCollate


class PrintAndSaveCallback(pl.callbacks.Callback):
    def __init__(self, store_prefix):
        super().__init__()
        self.store_prefix = store_prefix
        self._best_loss = 99999

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss = pl_module._val_loss
        sys.stdout.write('\n\n\tVal loss: {0}\n'.format(val_loss))
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

        # for lang in pl_module._epoch_results:
        #     res = pl_module._epoch_results[lang]
        #     if "acc_best" in res:
        #         trainer.save_checkpoint(self.store_prefix + "." + lang + ".best")
        #
        # trainer.save_checkpoint(self.store_prefix + ".last")
        #
        # s = "{0:30s}\tACC".format("Language")
        # print("\n\n\t" + s)
        # print("\t" + ("=" * (len(s) + 16)))
        # for lang in pl_module._language_codes:
        #     acc = metrics["val/ACC/{0}".format(lang)]
        #     msg = "\t{0:30s}:\t{1:.4f}".format(lang, acc)
        #     print(msg)
        # print("\n")


def _train(params):
    upsample = ''.join(params.upsample).replace('[', '').replace(']', '').split(',')
    upsample = [int(x) for x in upsample]
    print(upsample)
    config = {
        'num_layers': params.num_layers,
        'layer_size': params.layer_size,
        'psamples': params.psamples,
        'stride': params.stride,
        'upsample': upsample,
        'sample_rate': params.sample_rate,
        'output': params.output
    }
    conf_file = '{0}.yaml'.format(params.output_base)
    yaml.dump(config, open(conf_file, 'w'))
    sys.stdout.write('=================Config=================\n')
    sys.stdout.write(open(conf_file).read())
    sys.stdout.write('========================================\n\n')
    trainset = VocoderDataset(params.train_folder, target_sample_rate=params.sample_rate,
                              max_segment_size=params.maximum_segment_size, random_start=True)
    devset = VocoderDataset(params.dev_folder, target_sample_rate=params.sample_rate,
                            max_segment_size=params.maximum_segment_size, random_start=False)
    sys.stdout.write('==================Data==================\n')
    sys.stdout.write('Training files: {0}\n'.format(len(trainset)))
    sys.stdout.write('Validation files: {0}\n'.format(len(devset)))
    sys.stdout.write('========================================\n\n')
    sys.stdout.write('================Training================\n')
    collate = VocoderCollate()
    trainloader = DataLoader(trainset,
                             batch_size=params.batch_size,
                             num_workers=params.num_workers,
                             collate_fn=collate.collate_fn)
    devloader = DataLoader(devset,
                           batch_size=params.batch_size,
                           num_workers=params.num_workers,
                           collate_fn=collate.collate_fn)

    model = CubenetVocoder(num_layers=params.num_layers,
                           layer_size=params.layer_size,
                           psamples=params.psamples,
                           stride=params.stride,
                           upsample=upsample,
                           learning_rate=params.lr,
                           output=params.output)

    if params.resume:
        sys.stdout.write('Resuming from previous checkpoint\n')
        sys.stdout.flush()
        model.load('{0}.last'.format(params.output_base))

    if params.gpus == 0:
        acc = 'cpu'
    else:
        acc = 'gpu'
    trainer = pl.Trainer(
        gpus=params.gpus,
        accelerator=acc,
        max_epochs=-1,
        gradient_clip_val=5,
        callbacks=[PrintAndSaveCallback(params.output_base)]
    )

    trainer.fit(model, trainloader, devloader)


if __name__ == '__main__':
    parser = ArgumentParser(description='NLP-Cube Trainer Helper')
    parser.add_argument('--output-base', action='store', dest='output_base',
                        default='data/vocoder',
                        help='Where to store the model (default=data/vocoder)')
    parser.add_argument('--checkpoint', action='store', dest='checkpoint',
                        type=int, default=5000,
                        help='Checkpoint interval (default=5000 steps)')
    parser.add_argument('--batch-size', dest='batch_size', default=16,
                        type=int, help='Batch size (default=16)')
    parser.add_argument('--num-workers', dest='num_workers', default=4,
                        type=int, help='Batch size (default=4)')
    parser.add_argument('--maximum-segment-size', dest='maximum_segment_size', type=int,
                        default=16348, help='Maximum audio segment size - will be randomly selected (default=16348)')
    parser.add_argument('--gpus', dest='gpus', default=1, type=int,
                        help='How many gpus to use (default=1) - use 0 for CPU only training (very slow)')
    parser.add_argument('--train-folder', dest='train_folder', default='data/processed/train',
                        help='Location of training files (default=data/processed/train)')
    parser.add_argument('--dev-folder', dest='dev_folder', default='data/processed/dev',
                        help='Location of training files (default=data/processed/dev)')
    parser.add_argument('--sample-rate', dest='sample_rate', type=int, default=22050,
                        help='Number of parallel samples (default=22050)')
    parser.add_argument('--psamples', dest='psamples', type=int, default=16,
                        help='Number of parallel samples (default=16)')
    parser.add_argument('--stride', dest='stride', type=int, default=16,
                        help='Distance between the parallel samples (default=16)')
    parser.add_argument('--num-mixtures', dest='num_mixtures', default=3, type=int,
                        help='Number of mixtures to use when generating samples (default=3)')
    parser.add_argument('--layer-size', dest='layer_size', default=512, type=int,
                        help='LSTM layer size (default=512)')
    parser.add_argument('--num-layers', dest='num_layers', default=1, type=int,
                        help='Number of LSTM layers (default=1)')
    parser.add_argument('--upsample', dest='upsample', default=[2, 2, 2, 2], type=list,
                        help='Upsample layers (default=[2,2,2,2])')
    parser.add_argument('--lr', dest='lr', default=1e-4, type=float,
                        help='Learning rate (default=1e-4)')
    parser.add_argument('--output', dest='output', default='mol',
                        help='Output type (mol|gm|mulaw|beta) (default=mol)')

    parser.add_argument('--resume', dest='resume', action='store_true')

    args = parser.parse_args()

    _train(args)
