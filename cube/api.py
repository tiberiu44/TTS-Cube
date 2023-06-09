import os
import sys
import yaml
import torch
import numpy as np
import scipy.io
from pathlib import Path
import random

sys.path.append('')
from cube.io_utils.repository import download_model
from cube.networks.cubegan import Cubegan
from cube.io_utils.io_cubegan import CubeganEncodings, CubeganCollate
from cube.io_utils.io_text import Text2FeatBlizzard, Text2Feat
from cube.utils.hf import HFTokenizer


class TTSCube:
    def __init__(self, model_path: str, phonemizer_path: str):
        encodings = CubeganEncodings('{0}.encodings'.format(model_path))
        conf = yaml.load(open('{0}.yaml'.format(model_path)), yaml.Loader)
        cond_type = conf['conditioning']
        self._model = Cubegan(encodings, conditioning=cond_type, train=False)
        self._model.load('{0}.model'.format(model_path))
        self._collate = CubeganCollate(encodings, conditioning_type=cond_type)
        try:
            self._text2feat = Text2FeatBlizzard(phonemizer_path=phonemizer_path)
        except:
            self._text2feat = Text2Feat(phonemizer_path=phonemizer_path)
        self._model.eval()
        self._text2feat._phonemizer.eval()
        if cond_type.startswith('hf:'):
            self._hf_tok = HFTokenizer(cond_type.split(':')[-1])
        else:
            self._hf_tok = None

    @staticmethod
    def load(model_name: str):
        base_name = '{0}/.ttscube/models/{1}'.format(str(Path.home()), model_name)
        if not os.path.exists(base_name):
            os.makedirs(base_name, exist_ok=True)
            download_model(base_name, model_name)
        return TTSCube('{0}/cubegan'.format(base_name), '{0}/phonemizer'.format(base_name))

    def __call__(self, text, speaker='none'):
        with torch.no_grad():
            rez = {'meta': self._text2feat(text)}
            rez['meta']['speaker'] = speaker
            rez['pitch'] = np.zeros((100))
            rez['mgc'] = np.zeros((100, 80))
            rez['meta']['words_left'] = []
            rez['meta']['words_right'] = []
            rez['meta']['frame2phon'] = [0] * 100
            if self._hf_tok is not None:
                rez['meta']['words_hf'] = self._hf_tok(rez['meta']['words'])
                rez['meta']['words_left_hf'] = {'tok_ids': []}
                rez['meta']['words_right_hf'] = {'tok_ids': []}
            with torch.no_grad():
                X = self._collate.collate_fn([rez])
                for key in X:
                    if isinstance(X[key], torch.Tensor):
                        X[key] = X[key].to(self._model.get_device())
                audio = self._model.inference(X)
                audio = audio.detach().cpu().numpy().squeeze()
                audio = np.asarray(audio * 32767, dtype=np.int16)
        return audio


if __name__ == '__main__':
    model = TTSCube.load('blizzard2023-hf')
    audio = model('Bonjour! Je suis un system artificialle.', speaker='neb')
    scipy.io.wavfile.write('tmp.wav', 24000, audio)
