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
from cube.networks.modules import LinearNorm, ConvNorm, UpsampleNetR
from cube.networks.loss import MOLOutput, GaussianOutput, BetaOutput, MULAWOutput


class CubenetVocoder(pl.LightningModule):
    def __init__(self,
                 num_layers: int = 2,
                 layer_size: int = 512,
                 upsample=100,
                 upsample_low=10,
                 use_lowres=True,
                 learning_rate=1e-4,
                 output='mol'):
        super(CubenetVocoder, self).__init__()

        self._learning_rate = learning_rate
        self._upsample_mel = UpsampleNetR(upsample=upsample)
        self._upsample_lowres = UpsampleNetR(upsample=upsample_low)
        self._use_lowres = use_lowres
        if self._use_lowres:
            self._lowres_conv = nn.ModuleList()
            ic = 1
            for ii in range(3):
                self._lowres_conv.append(ConvNorm(ic, 20, kernel_size=3, padding=1))
                ic = 20
        ic = 80 + 1
        if use_lowres:
            ic += 20
        self._skip = LinearNorm(ic, layer_size, w_init_gain='tanh')
        rnn_list = []
        for ii in range(num_layers):
            rnn = nn.GRU(input_size=ic, hidden_size=layer_size, num_layers=1, batch_first=True)
            ic = layer_size
            rnn_list.append(rnn)
        self._rnns = nn.ModuleList(rnn_list)
        self._preoutput = LinearNorm(layer_size, 256)

        if output == 'mol':
            self._output_functions = MOLOutput()
        elif output == 'gm':
            self._output_functions = GaussianOutput()
        elif output == 'beta':
            self._output_functions = BetaOutput()
        elif output == 'mulaw':
            self._output_functions = MULAWOutput()

        self._output = LinearNorm(256, self._output_functions.sample_size, w_init_gain='linear')
        self._val_loss = 9999

    def forward(self, X):
        mel = X['mel']
        if 'x' in X:
            return self._train_forward(X)
        else:
            return self._inference(X)

    def _inference(self, X):
        with torch.no_grad():
            mel = X['mel']
            low_x = X['x_low']
            if self._use_lowres:
                # conv and upsample
                hidden = low_x.unsqueeze(1)
                for conv in self._lowres_conv:
                    hidden = torch.tanh(conv(hidden))

                upsampled_x = self._upsample_lowres(hidden).permute(0, 2, 1)

            upsampled_mel = self._upsample_mel(mel.permute(0, 2, 1)).permute(0, 2, 1)
            cond = upsampled_mel
            if self._use_lowres:
                msize = min(upsampled_mel.shape[1], upsampled_x.shape[1])
                cond = torch.cat([upsampled_mel[:, :msize, :], upsampled_x[:, :msize, :]], dim=-1)

            last_x = torch.ones((cond.shape[0], 1, 1),
                                device=self._get_device()) * 0  # * self._x_zero
            output_list = []
            hxs = [None for _ in range(len(self._rnns))]
            # index = 0
            for ii in tqdm.tqdm(range(cond.shape[1]), ncols=80):
                hidden = cond[:, ii, :].unsqueeze(1)
                res = self._skip(torch.cat([cond[:, ii, :].unsqueeze(1), last_x], dim=-1))
                hidden = torch.cat([hidden, last_x], dim=-1)
                for ll in range(len(self._rnns)):
                    rnn_input = hidden  # torch.cat([hidden, last_x], dim=-1)
                    rnn = self._rnns[ll]
                    rnn_output, hxs[ll] = rnn(rnn_input, hx=hxs[ll])
                    hidden = rnn_output
                    res = res + hidden

                preoutput = torch.tanh(self._preoutput(res))
                output = self._output(preoutput)
                output = output.reshape(output.shape[0], -1, self._output_functions.sample_size)
                samples = self._output_functions.sample(output)
                last_x = samples.unsqueeze(1)
                output_list.append(samples.unsqueeze(1))

        output_list = torch.cat(output_list, dim=1)
        return output_list.detach().cpu().numpy()  # self._output_functions.decode(output_list)

    def _train_forward(self, X):
        mel = X['mel']
        gs_x = X['x']
        low_x = X['x_low']

        # check if we are using lowres signal conditioning
        if self._use_lowres:
            # conv and upsample
            hidden = low_x.unsqueeze(1)
            for conv in self._lowres_conv:
                hidden = torch.tanh(conv(hidden))

            upsampled_x = self._upsample_lowres(hidden).permute(0, 2, 1)
        upsampled_mel = self._upsample_mel(mel.permute(0, 2, 1)).permute(0, 2, 1)

        msize = min(upsampled_mel.shape[1], gs_x.shape[1], upsampled_x.shape[1])
        upsampled_mel = upsampled_mel[:, :msize, :]
        gs_x = gs_x[:, :msize].unsqueeze(2)
        upsampled_x = upsampled_x[:, :msize, :]
        if self._use_lowres:
            hidden = torch.cat([upsampled_mel, upsampled_x, gs_x], dim=-1)
        else:
            hidden = torch.cat([upsampled_mel, gs_x], dim=-1)
        res = self._skip(hidden)

        for ll in range(len(self._rnns)):
            rnn_input = hidden
            rnn_output, _ = self._rnns[ll](rnn_input)
            hidden = rnn_output
            res = res + hidden
        preoutput = torch.tanh(self._preoutput(res))
        output = self._output(preoutput)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        gs_audio = batch['x']

        target_x = gs_audio[:, 1:]
        pred_x = output[:, :-1]
        loss = self._output_functions.loss(pred_x, target_x)
        return loss

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        gs_audio = batch['x']

        target_x = gs_audio[:, 1:]
        pred_x = output[:, :-1]
        loss = self._output_functions.loss(pred_x, target_x)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        loss = sum(outputs) / len(outputs)
        self.log("val_loss", loss)
        self._val_loss = loss

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
    fname = 'data/voc-anca-2-256-mulaw'
    conf = yaml.load(open('{0}.yaml'.format(fname)), Loader)
    num_layers = conf['num_layers']
    hop_size = conf['hop_size']
    layer_size = conf['layer_size']
    sample_rate = conf['sample_rate']
    sample_rate_low = conf['sample_rate_low']
    use_lowres = conf['use_lowres']
    output = conf['output']
    upsample = conf['upsample']
    vocoder = CubenetVocoder(num_layers=num_layers,
                             layer_size=layer_size,
                             use_lowres=use_lowres,
                             upsample=upsample,
                             upsample_low=sample_rate // sample_rate_low,
                             output=output)
    # vocoder = CubenetVocoder(num_layers=1, layer_size=1024)
    vocoder.load('{0}.last'.format(fname))
    import librosa
    from cube.io_utils.vocoder import MelVocoder
    from cube.io_utils.dataset import DatasetIO

    dio = DatasetIO()

    wav, sr = librosa.load('data/test.wav', sr=sample_rate)
    wav_low, sr = librosa.load('data/test.wav', sr=sample_rate_low)
    mel_vocoder = MelVocoder()
    mel = mel_vocoder.melspectrogram(wav, sample_rate=22050, hop_size=hop_size, num_mels=80, use_preemphasis=False)
    mel = torch.tensor(mel).unsqueeze(0)
    x_low = torch.tensor(wav_low).unsqueeze(0)
    dio.write_wave("data/load.wav", x_low.squeeze() * 32000, sample_rate_low, dtype=np.int16)
    vocoder.eval()
    start = time.time()
    # normalize mel
    output = vocoder({'mel': mel, 'x_low': x_low})
    # from ipdb import set_trace

    # set_trace()
    stop = time.time()
    print("generated {0} seconds of audio in {1}".format(len(wav) / sample_rate, stop - start))

    dio.write_wave("data/generated.wav", output.squeeze() * 32000, sample_rate, dtype=np.int16)
