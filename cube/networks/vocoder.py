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

import torch
import torch.nn as nn
import pytorch_lightning as pl

import sys

sys.path.append('')

from cube.networks.modules import LinearNorm, ConvNorm, UpsampleNet
from cube.networks.loss import gaussian_loss


class CubenetVocoder(pl.LightningModule):
    def __init__(self, num_layers: int = 2, layer_size: int = 512, psamples: int = 16, stride: int = 16, upsample=256):
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
        self._rnn = nn.LSTM(input_size=80 + psamples, hidden_size=layer_size, num_layers=num_layers, batch_first=True)
        self._output = LinearNorm(layer_size, psamples * 2)  # mean+logvars
        self._loss = gaussian_loss

    def forward(self, X):
        mel = X['mel']
        if 'x' in X:
            return self._train_forward(mel, X['x'])
        else:
            return self._inference(mel)

    def _train_forward(self, mel, gs_audio):

        from ipdb import set_trace
        upsampled_mel = self._upsample(mel.permute(0, 2, 1)).permute(0, 2, 1)
        # upsampled_mel = torch.repeat_interleave(upsampled_mel, self._stride, dim=1)
        # get closest gs_size that is multiple of stride
        x_size = ((gs_audio.shape[1] // (self._stride * self._psamples)) + 1) * self._stride * self._psamples
        x = nn.functional.pad(gs_audio, (0, x_size - gs_audio.shape[1]))
        x = x.reshape(x.shape[0], x.shape[1] // self._stride, -1)
        x = x.reshape(x.shape[0], -1, self._stride, self._psamples)
        x = x.transpose(2, 3)
        x = x.reshape(x.shape[0], -1, self._psamples)
        rnn_input = torch.cat([upsampled_mel, x], dim=-1)
        rnn_output, _ = self._rnn(rnn_input)
        return self._output(rnn_output)

    def _inference(self, mel):
        pass

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        from ipdb import set_trace
        set_trace()
        loss = self._loss(output, batch['x'])
        return loss

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self._loss(output, batch['x'])
        return loss

    def validation_epoch_end(self, outputs) -> None:
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    @torch.jit.ignore
    def _get_device(self):
        if self.input_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self.input_emb.weight.device.type, str(self.input_emb.weight.device.index))

    @torch.jit.ignore
    def save(self, path):
        torch.save(self.state_dict(), path)

    @torch.jit.ignore
    def load(self, path):
        # from ipdb import set_trace
        # set_trace()
        # tmp = torch.load(path, map_location='cpu')
        self.load_state_dict(torch.load(path, map_location='cpu'))
