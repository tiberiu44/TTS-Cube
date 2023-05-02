import time

import torch
import torch.nn as nn
import numpy as np
import scipy.io

import sys
import torchaudio
import torchaudio.transforms as T

sys.path.append('')
from cube.networks.enhancer import Cubedall
from argparse import ArgumentParser


def _enhance(params):
    model = Cubedall(lr=1e-10)
    sys.stdout.write('Loading model checkpoint\n')
    sys.stdout.flush()
    model.load('{0}.last'.format(params.model_base))
    model.eval()
    audio, sample_rate = torchaudio.load(params.input_file)
    res = T.Resample(sample_rate, 48000, dtype=audio.dtype)
    audio = res(audio)
    sys.stdout.write(f'Source sample rate is {sample_rate}\n')
    start = time.time()
    sys.stdout.write('Enhancing...')
    sys.stdout.flush()
    with torch.no_grad():
        x = {
            'x': audio,
            'denoise': torch.ones((1, 1))
        }
        enhanced_audio = model.inference(x)
    stop = time.time()
    sys.stdout.write(f' done in {stop - start} seconds\n')
    print(audio.shape)
    print(enhanced_audio.shape)
    audio = enhanced_audio.detach().cpu().numpy().squeeze()
    audio = np.asarray(audio * 32767, dtype=np.int16)
    scipy.io.wavfile.write(params.output_file, 48000, audio)


if __name__ == '__main__':
    parser = ArgumentParser(description='NLP-Cube Trainer Helper')
    parser.add_argument('--model-base', action='store', dest='model_base',
                        default='data/cubedall',
                        help='Where to store the model (default=data/cubedall)')
    parser.add_argument('--input-file', action='store', dest='input_file',
                        help='Input file to enhance')
    parser.add_argument('--output-file', action='store', dest='output_file',
                        help='Input file to enhance')

    args = parser.parse_args()

    if args.input_file and args.output_file:
        _enhance(args)
    else:
        parser.print_help()
