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
import time

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

import sys

import tqdm
import yaml

sys.path.append('')
from yaml import Loader
from cube.networks.modules import WaveRNN


class CubenetVocoder(pl.LightningModule):
    def __init__(self,
                 num_layers_lr: int = 2,
                 layer_size_lr: int = 512,
                 num_layers_hr: int = 2,
                 layer_size_hr: int = 512,
                 upsample=100,
                 upsample_low=10,
                 learning_rate=1e-4,
                 output='mol'):
        super(CubenetVocoder, self).__init__()
        self._learning_rate = learning_rate
        self._wavernn_hr = WaveRNN(num_layers=num_layers_hr,
                                   layer_size=layer_size_hr,
                                   upsample=upsample,
                                   use_lowres=True,
                                   upsample_low=upsample_low,
                                   learning_rate=learning_rate,
                                   output=output)
        self._wavernn_lr = WaveRNN(num_layers=num_layers_lr,
                                   layer_size=layer_size_lr,
                                   upsample=upsample // upsample_low,
                                   use_lowres=False,
                                   learning_rate=learning_rate,
                                   output=output)
        self._val_loss_hr = 9999
        self._val_loss_lr = 9999

    def forward(self, X):
        if 'x' in X:
            return self._train(X)
        else:
            return self._inference(X)

    def _train(self, X):
        x = X['x']
        x_low = X['x_low']
        mel = X['mel']
        loss_hr = self._wavernn_hr.training_step(
            {
                'x': x,
                'x_low': x_low,
                'mel': mel
            },
            batch_idx=-1
        )
        loss_lr = self._wavernn_lr.training_step(
            {
                'x': x_low,
                'mel': mel
            },
            batch_idx=-1
        )
        return {
            'lr': loss_lr,
            'hr': loss_hr,
            'loss': (loss_hr + loss_lr) / 2
        }

    def _inference(self, X):
        with torch.no_grad():
            x_lr = self._wavernn_lr(X)
            x_hr = self._wavernn_hr(
                {
                    'mel': X['mel'],
                    # 'x_low': torch.tensor(x_lr).squeeze().unsqueeze(0)
                    'x_low': X['x_low']
                }
            )
        return x_lr, x_hr

    def validation_step(self, batch, batch_idx):
        return self.forward(batch)

    def training_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_epoch_end(self, outputs) -> None:
        loss_lr = sum([x['lr'] for x in outputs]) / len(outputs)
        loss_hr = sum([x['hr'] for x in outputs]) / len(outputs)
        self.log("val_loss", (loss_hr + loss_lr) / 2)
        self.log("val_loss_lr", loss_lr)
        self.log("val_loss_hr", loss_hr)
        self._val_loss_hr = loss_hr
        self._val_loss_lr = loss_lr

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._learning_rate)

    @torch.jit.ignore
    def _get_device(self):
        if self._output.linear_layer.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._output.linear_layer.weight.device.type,
                                str(self._output.linear_layer.weight.device.index))

    @torch.jit.ignore
    def save(self, path):
        torch.save(self.state_dict(), path)

    @torch.jit.ignore
    def load(self, path):
        # from ipdb import set_trace
        # set_trace()
        # tmp = torch.load(path, map_location='cpu')
        self.load_state_dict(torch.load(path, map_location='cpu'))


if __name__ == '__main__':
    fname = 'data/voc-anca-1-512-mol'
    conf = yaml.load(open('{0}.yaml'.format(fname)), Loader)
    num_layers_hr = conf['num_layers_hr']
    layer_size_hr = conf['layer_size_hr']
    num_layers_lr = conf['num_layers_lr']
    layer_size_lr = conf['layer_size_lr']
    hop_size = conf['hop_size']
    sample_rate = conf['sample_rate']
    sample_rate_low = conf['sample_rate_low']
    output = conf['output']
    upsample = conf['upsample']
    vocoder = CubenetVocoder(num_layers_hr=num_layers_hr,
                             layer_size_hr=layer_size_hr,
                             layer_size_lr=layer_size_lr,
                             num_layers_lr=num_layers_lr,
                             upsample=upsample,
                             upsample_low=sample_rate // sample_rate_low,
                             output=output)
    # vocoder = CubenetVocoder(num_layers=1, layer_size=1024)
    vocoder.load('{0}.last'.format(fname))
    vocoder._wavernn_lr.load('{0}.lr.best'.format(fname))
    vocoder._wavernn_hr.load('{0}.hr.best'.format(fname))
    import librosa
    from cube.io_utils.vocoder import MelVocoder
    from cube.io_utils.dataset import DatasetIO

    dio = DatasetIO()

    wav, sr = librosa.load('data/test.wav', sr=sample_rate)
    from cube.networks.loss import MULAWOutput

    wav2 = MULAWOutput().decode(MULAWOutput().encode(wav))
    dio.write_wave("data/mulaw.wav", wav2 * 32000, sample_rate, dtype=np.int16)

    wav_low, sr = librosa.load('data/test.wav', sr=sample_rate_low)
    mel_vocoder = MelVocoder()
    mel = mel_vocoder.melspectrogram(wav, sample_rate=sample_rate, hop_size=hop_size, num_mels=80,
                                     use_preemphasis=False)
    mel = torch.tensor(mel).unsqueeze(0)
    x_low = torch.tensor(wav_low).unsqueeze(0)
    # dio.write_wave("data/load.wav", x_low.squeeze() * 32000, sample_rate_low, dtype=np.int16)
    vocoder.eval()
    start = time.time()
    # normalize mel
    output_lr, output_hr = vocoder({'mel': mel, 'x_low': x_low})
    # from ipdb import set_trace

    # set_trace()
    stop = time.time()
    print("generated {0} seconds of audio in {1}".format(len(wav) / sample_rate, stop - start))

    dio.write_wave("data/generated-lr.wav", output_lr.squeeze() * 32000, sample_rate_low, dtype=np.int16)
    dio.write_wave("data/generated-hr.wav", output_hr.squeeze() * 32000, sample_rate, dtype=np.int16)
