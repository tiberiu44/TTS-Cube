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


def _train(params):
    config = {
        'num_layers': params.num_layers,
        'layer_size': params.layer_size,
        'psamples': params.psamples,
        'stride': params.stride,
        'upsample': 256,
        'sample_rate': params.sample_rate
    }
    conf_file = '{0}.yaml'.format(params.output_base)
    yaml.dump(config, open(conf_file, 'w'))
    sys.stdout.write('=================Config=================\n')
    sys.stdout.write(open(conf_file).read())
    sys.stdout.write('========================================\n\n')
    trainset = VocoderDataset(params.train_folder, target_sample_rate=params.sample_rate,
                              max_segment_size=params.maximum_segment_size)
    devset = VocoderDataset(params.dev_folder, target_sample_rate=params.sample_rate)
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
                           upsample=256)

    if params.gpus == 0:
        acc = 'cpu'
    else:
        acc = 'gpu'
    trainer = pl.Trainer(
        gpus=params.gpus,
        accelerator=acc,
        gradient_clip_val=5
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
    parser.add_argument('--num_layers', dest='num_layers', default=2, type=int,
                        help='Number of LSTM layers (default=2)')

    args = parser.parse_args()

    _train(args)
