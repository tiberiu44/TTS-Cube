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
from cube.networks.modules import LinearNorm, UpsampleNet
from cube.networks.loss import beta_loss


class CubenetVocoder(pl.LightningModule):
    def __init__(self, num_layers: int = 2, layer_size: int = 512, psamples: int = 16, stride: int = 16,
                 upsample=[2, 2, 2, 2]):
        super(CubenetVocoder, self).__init__()

        self._config = {
            'num_layers': num_layers,
            'layer_size': layer_size,
            'psamples': psamples,
            'stride': stride,
            'upsample': upsample
        }
        self._stride = stride
        self._psamples = psamples
        self._upsample = UpsampleNet(upsample_scales=upsample, in_channels=80, out_channels=80)
        # self._rnn = nn.GRU(input_size=80 + psamples, hidden_size=layer_size, num_layers=num_layers, batch_first=True)
        ic = 80
        rnn_list = []
        for ii in range(num_layers):
            rnn = nn.GRU(input_size=ic + psamples, hidden_size=layer_size, num_layers=1, batch_first=True)
            ic = layer_size
            rnn_list.append(rnn)
        self._rnns = nn.ModuleList(rnn_list)
        self._preoutput = LinearNorm(layer_size, 256)
        self._skip = LinearNorm(80, layer_size)
        self._output = LinearNorm(256, psamples * 2)  # mean+logvars
        self._output_aux = LinearNorm(80, psamples * 2)
        self._loss = beta_loss
        self._val_loss = 9999

    def forward(self, X):
        mel = X['mel']
        if 'x' in X:
            return self._train_forward(mel, X['x'])
        else:
            return self._inference(mel)

    def _inference(self, mel):
        with torch.no_grad():
            upsampled_mel = self._upsample(mel.permute(0, 2, 1)).permute(0, 2, 1)
            last_x = torch.zeros((upsampled_mel.shape[0], 1, self._psamples), device=self._get_device())
            output_list = []
            hxs = [None for _ in range(len(self._rnns))]
            # index = 0
            for ii in tqdm.tqdm(range(upsampled_mel.shape[1]), ncols=80):
                hidden = upsampled_mel[:, ii, :].unsqueeze(1)
                res = self._skip(upsampled_mel[:, ii, :].unsqueeze(1))
                for ll in range(len(self._rnns)):
                    rnn_input = torch.cat([hidden, last_x], dim=-1)
                    rnn = self._rnns[ll]
                    rnn_output, hxs[ll] = rnn(rnn_input, hx=hxs[ll])
                    hidden = rnn_output + res
                    res = hidden

                preoutput = torch.tanh(self._preoutput(res))
                output = self._output(preoutput)
                output = output.reshape(output.shape[0], -1, 2)
                means = output[:, :, 0]
                logvars = output[:, :, 1]
                z = torch.randn((output.shape[0], output.shape[1]), device=self._get_device()) * 0.8
                samples = means + z * torch.exp(logvars)
                last_x = samples.unsqueeze(1)
                output_list.append(samples.unsqueeze(1))

        output_list = torch.cat(output_list, dim=1)
        output_list = output_list.reshape(output_list.shape[0], -1, self._stride, self._psamples)
        output_list = output_list.transpose(2, 3)
        output_list = output_list.reshape(output_list.shape[0], -1).detach().cpu().numpy()
        return output_list

    def _train_forward(self, mel, gs_audio):

        upsampled_mel = self._upsample(mel.permute(0, 2, 1)).permute(0, 2, 1)
        skip = self._skip(upsampled_mel)
        # get closest gs_size that is multiple of stride
        x_size = ((gs_audio.shape[1] // (self._stride * self._psamples)) + 1) * self._stride * self._psamples
        x = nn.functional.pad(gs_audio, (0, x_size - gs_audio.shape[1]))
        x = x.reshape(x.shape[0], -1, self._stride, self._psamples)
        x = x.transpose(2, 3)
        x = x.reshape(x.shape[0], -1, self._psamples)
        msize = min(upsampled_mel.shape[1], x.shape[1])
        hidden = upsampled_mel[:, :msize, :]
        last_x = x[:, :msize, :]
        res = skip[:, :msize, :]
        for ll in range(len(self._rnns)):
            rnn_input = torch.cat([hidden, last_x], dim=-1)
            rnn_output, _ = self._rnns[ll](rnn_input)
            hidden = rnn_output + res
            res = hidden

        preoutput = torch.tanh(self._preoutput(res))
        output = self._output(preoutput)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        gs_audio = batch['x']
        x_size = ((gs_audio.shape[1] // (self._stride * self._psamples)) + 1) * self._stride * self._psamples
        x = nn.functional.pad(gs_audio, (0, x_size - gs_audio.shape[1] + 1))
        x = x[:, 1:]
        x = x.reshape(x.shape[0], x.shape[1] // self._stride, -1)
        x = x.reshape(x.shape[0], -1, self._stride, self._psamples)
        x = x.transpose(2, 3)
        target_x = x.reshape(x.shape[0], -1, self._psamples)
        output = output.reshape(output.shape[0], -1, 2)
        target_x = target_x.reshape(target_x.shape[0], -1)
        loss = self._loss(output[:, self._psamples * self._stride:, ],
                          target_x[:, self._psamples * self._stride:],
                          log_std_min=-14)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        gs_audio = batch['x']
        x_size = ((gs_audio.shape[1] // (self._stride * self._psamples)) + 1) * self._stride * self._psamples
        x = nn.functional.pad(gs_audio, (0, x_size - gs_audio.shape[1] + 1))
        x = x[:, 1:]
        x = x.reshape(x.shape[0], x.shape[1] // self._stride, -1)
        x = x.reshape(x.shape[0], -1, self._stride, self._psamples)
        x = x.transpose(2, 3)
        target_x = x.reshape(x.shape[0], -1, self._psamples)
        output = output.reshape(output.shape[0], -1, 2)
        target_x = target_x.reshape(target_x.shape[0], -1)
        loss = self._loss(output[:, self._psamples * self._stride:, ],
                          target_x[:, self._psamples * self._stride:],
                          log_std_min=-14)
        return loss.mean()

    def validation_epoch_end(self, outputs) -> None:
        loss = sum(outputs) / len(outputs)
        self.log("val_loss", loss)
        self._val_loss = loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

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
    fname = 'data/voc-anca-16-16'
    conf = yaml.load(open('{0}.yaml'.format(fname)), Loader)
    num_layers = conf['num_layers']
    upsample = conf['upsample']
    psamples = conf['psamples']
    stride = conf['stride']
    layer_size = conf['layer_size']
    sample_rate = conf['sample_rate']
    vocoder = CubenetVocoder(num_layers=num_layers,
                             layer_size=layer_size,
                             psamples=psamples,
                             stride=stride,
                             upsample=upsample)
    # vocoder = CubenetVocoder(num_layers=1, layer_size=1024)
    vocoder.load('{0}.last'.format(fname))
    import librosa
    from cube.io_utils.vocoder import MelVocoder

    wav, sr = librosa.load('data/test.wav', sr=22050)
    mel_vocoder = MelVocoder()
    mel = mel_vocoder.melspectrogram(wav, sample_rate=22050, num_mels=80, use_preemphasis=False)
    mel = torch.tensor(mel).unsqueeze(0)
    vocoder.eval()
    start = time.time()
    output = vocoder({'mel': mel})
    # from ipdb import set_trace

    # set_trace()
    stop = time.time()
    print("generated {0} seconds of audio in {1}".format(len(wav) / 22050, stop - start))
    from cube.io_utils.dataset import DatasetIO

    dio = DatasetIO()

    dio.write_wave("data/generated.wav", output.squeeze() * 32000, 22050, dtype=np.int16)
