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
        self.automatic_optimization = False
        self._global_step = 0
        self._upsample = upsample
        self._upsample_low = upsample_low

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

            # x_low = X['x_low']  # torch.tensor(x_lr)
            x_low = torch.tensor(x_lr)
            inference_batch = self._inference_batch(X['mel'], x_low, num_batches=20)
            batched_x_hr = self._wavernn_hr(
                inference_batch
            )
            x_hr = self._compose_batched_inference(batched_x_hr)
        return x_lr, x_hr

    def _compose_batched_inference(self, batched_x):
        batched_x = batched_x[:, self._upsample:]
        return batched_x.reshape(1, -1)

    def _inference_batch(self, mel, x_low, num_batches=5):
        if mel.shape[1] < num_batches:
            num_batches = mel.shape[1]
        mel = mel[:, :mel.shape[1] // num_batches * num_batches]

        x_low = x_low[:, :x_low.shape[1] // num_batches * num_batches]
        mel_split = mel.reshape(num_batches, -1, mel.shape[2]).cpu().numpy()
        x_low_split = x_low.reshape(num_batches, -1).cpu().numpy()
        mel = np.ones((mel_split.shape[0], mel_split.shape[1] + 1, mel_split.shape[2])) * -5
        mel[:, 1:, :] = mel_split
        mel[1:, 0, :] = mel_split[:-1, -1, :]
        x_low = np.zeros((x_low_split.shape[0], x_low_split.shape[1] + self._upsample_low))
        x_low[:, self._upsample_low:] = x_low_split
        x_low[1:, 0:self._upsample_low] = x_low_split[:-1, -self._upsample_low:]

        return {
            'mel': torch.tensor(mel, dtype=torch.float),
            'x_low': torch.tensor(x_low, dtype=torch.float)
        }

    def validation_step(self, batch, batch_idx):
        return self.forward(batch)

    def training_step(self, batch, batch_idx):
        opt_lr, opt_hr = self.optimizers()
        opt_lr.zero_grad()
        opt_hr.zero_grad()

        loss = self.forward(batch)
        loss_lr = loss['lr']
        loss_hr = loss['hr']
        loss_lr.backward()
        loss_hr.backward()
        torch.nn.utils.clip_grad_norm(self._wavernn_lr.parameters(), 5)
        torch.nn.utils.clip_grad_norm(self._wavernn_hr.parameters(), 5)
        opt_lr.step()
        opt_hr.step()
        self._global_step += 1
        alpha = self._compute_lr(self._learning_rate, 5e-5, self._global_step)
        opt_lr.param_groups[0]['lr'] = alpha
        opt_hr.param_groups[0]['lr'] = alpha
        loss['alpha'] = alpha
        self.log_dict(loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        loss_lr = sum([x['lr'] for x in outputs]) / len(outputs)
        loss_hr = sum([x['hr'] for x in outputs]) / len(outputs)
        self.log("val_loss", (loss_hr + loss_lr) / 2)
        self.log("val_loss_lr", loss_lr)
        self.log("val_loss_hr", loss_hr)
        self._val_loss_hr = loss_hr
        self._val_loss_lr = loss_lr

    def configure_optimizers(self):
        return [
            torch.optim.Adam(self._wavernn_lr.parameters(), lr=self._learning_rate),
            torch.optim.Adam(self._wavernn_hr.parameters(), lr=self._learning_rate)
        ]
        # return torch.optim.Adam(self.parameters(), lr=self._learning_rate, amsgrad=True)

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
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def _compute_lr(self, initial_lr, delta, step):
        return initial_lr / (1 + delta * step)


if __name__ == '__main__':
    fname = 'data/voc-blizzard-neb-2-512-sg'
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
    # import torch.quantization
    #
    # # set quantization config for server (x86)
    # vocoder.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    #
    # # insert observers
    # torch.quantization.prepare(vocoder, inplace=True)
    # # Calibrate the model and collect statistics
    #
    # # convert to quantized version
    # torch.quantization.convert(vocoder, inplace=True)

    import librosa
    from cube.io_utils.vocoder import MelVocoder
    from cube.io_utils.dataset import DatasetIO

    dio = DatasetIO()

    wav, sr = librosa.load('data/test2.wav', sr=sample_rate)
    wav = wav / np.max(np.abs(wav)) * 0.98
    from cube.networks.loss import MULAWOutput

    wav2 = MULAWOutput().decode(MULAWOutput().encode(wav))
    dio.write_wave("data/mulaw.wav", wav2 * 32767, sample_rate, dtype=np.int16)
    dio.write_wave("data/norm.wav", wav * 32767, sample_rate, dtype=np.int16)

    wav_low, sr = librosa.load('data/test2.wav', sr=sample_rate_low)
    wav_low = wav_low / np.max(np.abs(wav_low)) * 0.98
    mel_vocoder = MelVocoder()
    mel = mel_vocoder.melspectrogram(wav, sample_rate=sample_rate, hop_size=hop_size, num_mels=80,
                                     use_preemphasis=False)
    mel = torch.tensor(mel, dtype=torch.float).unsqueeze(0)
    x_low = torch.tensor(wav_low).unsqueeze(0)
    # dio.write_wave("data/load.wav", x_low.squeeze() * 32000, sample_rate_low, dtype=np.int16)
    vocoder.eval()
    start = time.time()
    # normalize mel
    x_low_hi = vocoder._wavernn_hr._upsample_lowres_i(x_low.unsqueeze(1))
    dio.write_wave("data/high.wav", x_low_hi.detach().cpu().numpy().squeeze() * 32767, sample_rate, dtype=np.int16)
    output_lr, output_hr = vocoder({'mel': mel, 'x_low': x_low})

    stop = time.time()
    print("generated {0} seconds of audio in {1}".format(len(wav) / sample_rate, stop - start))

    dio.write_wave("data/generated-lr.wav", output_lr.squeeze() * 32767, sample_rate_low, dtype=np.int16)
    dio.write_wave("data/generated-hr.wav", output_hr.squeeze() * 32767, sample_rate, dtype=np.int16)
