import torch
import torch.nn as nn
import sys
import os
import json
import numpy as np
from PIL import Image
import yaml
from yaml import Loader

import tqdm

sys.path.append('')
sys.path.append('hifigan')

from cube.networks.cubegan import Cubegan
from cube.io_utils.io_cubegan import CubeganEncodings, CubeganCollate, CubeganDataset
from cube.io_utils.io_enhancer import EnhancerDataset, collate_fn
from hifigan.models import Generator
from hifigan.env import AttrDict
import scipy.io


def render_spectrogram(mgc, output_file):
    bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
    mgc_min = mgc.min()
    mgc_max = mgc.max()

    for x in range(mgc.shape[0]):
        for y in range(mgc.shape[1]):
            val = (mgc[x, y] - mgc_min) / (mgc_max - mgc_min)

            color = val * 255
            bitmap[mgc.shape[1] - y - 1, x] = [color, color, color]

    img = Image.fromarray(bitmap)  # smp.toimage(bitmap)
    img.save(output_file)


def cubegan_synthesize_dataset(model: Cubegan, output_path, devset_path, limit=-1, free=True, conditioning=None,
                               speaker=None):
    enc = model._encodings
    collate = CubeganCollate(enc, conditioning_type=conditioning)
    # load validation set
    hf_model = None
    if conditioning is not None and conditioning.startswith('hf'):
        hf_model = conditioning.split(':')[-1]
    dataset = CubeganDataset(devset_path, hf_model=hf_model)
    m_gen = len(dataset)
    if limit != -1 and limit < m_gen:
        m_gen = limit
    with torch.no_grad():
        for ii in tqdm.tqdm(range(m_gen)):
            if speaker is not None:
                dataset[ii]['meta']['speaker'] = 'neb'
            X = collate.collate_fn([dataset[ii]])
            for key in X:
                if isinstance(X[key], torch.Tensor):
                    X[key] = X[key].to(model.get_device())
            if free:
                audio = model.inference(X)
            else:
                audio = model(X)
            audio = audio.detach().cpu().numpy().squeeze()
            audio = np.asarray(audio * 32767, dtype=np.int16)
            scipy.io.wavfile.write('{0}/{1}.wav'.format(output_path, dataset[ii]['meta']['id']), 24000, audio)


def cubedall_synthesize(model: Cubegan, output_path, devset_path, limit=-1):
    dataset = EnhancerDataset(devset_path)
    m_gen = len(dataset)
    if limit != -1 and limit < m_gen:
        m_gen = limit
    index = 0
    with torch.no_grad():
        for ii in tqdm.tqdm(range(m_gen)):
            X = collate_fn([dataset[ii]])
            for key in X:
                if isinstance(X[key], torch.Tensor):
                    X[key] = X[key].to(model.get_device())
            audio = model(X)

            audio = audio.detach().cpu().numpy().squeeze()
            audio = np.asarray(audio * 32767, dtype=np.int16)
            scipy.io.wavfile.write('{0}/{1}.out.wav'.format(output_path, index), 48000, audio)
            audio = np.asarray(X['x'].squeeze().cpu().numpy() * 32767, dtype=np.int16)
            scipy.io.wavfile.write('{0}/{1}.inp.wav'.format(output_path, index), 48000, audio)
            audio = np.asarray(X['y'].squeeze().cpu().numpy() * 32767, dtype=np.int16)
            scipy.io.wavfile.write('{0}/{1}.tar.wav'.format(output_path, index), 48000, audio)
            index += 1


if __name__ == '__main__':
    encodings = CubeganEncodings()
    encodings.load('data/cubenet-multi-bert.encodings')
    import yaml

    conf = yaml.load(open('data/cubenet-multi-bert.yaml'), Loader)
    model = Cubegan(encodings, conditioning=conf['conditioning'])
    model.load('data/cubenet-multi-bert.last')
    model.eval()
    cubegan_synthesize_dataset(model, 'generated_files/forced/tmp/', 'data/processed/dev', free=False,
                               conditioning=conf['conditioning'], speaker='neb')
    cubegan_synthesize_dataset(model, 'generated_files/free/tmp/', 'data/processed/dev', free=True,
                               conditioning=conf['conditioning'], speaker='neb')
    # synthesize_devset('data/textcoder-neb-baseline',
    #                   'data/models/vocoder/neb-noft/g_00600000',
    #                   output_path='generated_files/free/',
    #                   forced_synthesis=False)
    # synthesize_devset('data/textcoder-neb-baseline',
    #                   'data/models/vocoder/neb-noft/g_00600000',
    #                   output_path='generated_files/forced/',
    #                   forced_synthesis=True)
