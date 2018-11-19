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

import numpy as np
import sys
from io_modules.dataset import DatasetIO
from io_modules.vocoder import MelVocoder
import torch
import torch.nn as nn

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BeeCoder:
    def __init__(self, params, model=None, runtime=False):
        self.params = params

        self.UPSAMPLE_COUNT = int(12.5 * params.target_sample_rate / 1000)
        self.RECEPTIVE_SIZE = 1024  # this means 64ms

        self.sparse = False
        self.dio = DatasetIO()
        self.vocoder = MelVocoder()

        self.network = VocoderNetwork(receptive_field=self.RECEPTIVE_SIZE).to(device)
        self.trainer = torch.optim.Adam(self.network.parameters(), lr=self.params.learning_rate)
        self.abs_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCELoss()
        self.cross_loss = torch.nn.CrossEntropyLoss()
        self.cnt = 0

    def synthesize(self, mgc, batch_size, sample=True, temperature=1.0, path=None):
        last_proc = 0
        synth = [0 for ii in range(self.RECEPTIVE_SIZE)]
        x = []
        for mgc_index in range(len(mgc)):
            curr_proc = int((mgc_index + 1) * 100 / len(mgc))
            if curr_proc % 5 == 0 and curr_proc != last_proc:
                while last_proc < curr_proc:
                    last_proc += 5
                    sys.stdout.write(' ' + str(last_proc))
                    sys.stdout.flush()

            input = mgc[mgc_index]

            x = [input for ii in range(self.UPSAMPLE_COUNT)]

            [signal, softmax] = self.network(x, prev=synth)
            #
            for zz in signal:
                synth.append(zz.item())
            x = []

        # synth = self.dio.ulaw_decode(synth, discreete=False)
        synth = np.array(synth[self.RECEPTIVE_SIZE:], dtype=np.float32)
        synth = np.clip(synth * 32768, -32767, 32767)
        synth = np.array(synth, dtype=np.int16)

        return synth

    def store(self, output_base):
        torch.save(self.network.state_dict(), output_base + ".network")
        # self.model.save(output_base + ".network")
        x = 0

    def load(self, output_base):
        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1:
                self.network.load_state_dict(torch.load(output_base + ".network", map_location='cuda:0'))
            else:
                self.network.load_state_dict(torch.load(output_base + ".network"))
        else:
            self.network.load_state_dict(
                torch.load(output_base + '.network', map_location=lambda storage, loc: storage))
        self.network.to(device)
        # self.model.populate(output_base + ".network")

    def _predict_one(self, mgc, noise):

        return None

    def _get_loss(self, signal_orig, signal_pred):
        loss = 0
        return loss

    def learn(self, wave, mgc, batch_size):
        last_proc = 0
        total_loss = 0
        num_batches = 0
        batch_size = batch_size * self.UPSAMPLE_COUNT
        mgc_list = []
        signal = [0 for ii in range(self.RECEPTIVE_SIZE)]
        for mgc_index in range(len(mgc)):
            curr_proc = int((mgc_index + 1) * 100 / len(mgc))
            if curr_proc % 5 == 0 and curr_proc != last_proc:
                while last_proc < curr_proc:
                    last_proc += 5
                    sys.stdout.write(' ' + str(last_proc))
                    sys.stdout.flush()
            if mgc_index < len(mgc) - 1:
                for ii in range(self.UPSAMPLE_COUNT):
                    mgc_list.append(mgc[mgc_index])
                    signal.append(wave[mgc_index * self.UPSAMPLE_COUNT + ii])

                if len(mgc_list) == batch_size:
                    self.trainer.zero_grad()
                    num_batches += 1
                    y_pred, y_softmax = self.network(mgc_list, signal=signal)
                    disc, cont = self.dio.ulaw_encode(signal[self.RECEPTIVE_SIZE:])
                    # from ipdb import set_trace
                    # set_trace()
                    y_target = torch.tensor(disc, dtype=torch.long).to(device)

                    loss = self.cross_loss(y_softmax, y_target)
                    total_loss += loss
                    loss.backward()
                    self.trainer.step()

                    mgc_list = []
                    signal = signal[-self.RECEPTIVE_SIZE:]

        total_loss = total_loss.item()
        # self.cnt += 1
        return total_loss / num_batches


class VocoderNetwork(nn.Module):
    def __init__(self, receptive_field=1024):
        super(VocoderNetwork, self).__init__()

        self.RECEPTIVE_FIELD = receptive_field
        self.NUM_NETWORKS = 1

        self.convolutions = nn.ModuleList([nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.Conv1d(32, 256, kernel_size=5, stride=2, padding=0),
            nn.Tanh()) for ii in range(self.NUM_NETWORKS)]

        )

        self.conditioning = nn.Sequential(
            nn.Linear(60, 200),
            nn.Tanh(),
            nn.Linear(200, 256),
            nn.Sigmoid()
        )

        for ii in range(self.NUM_NETWORKS):
            torch.nn.init.xavier_uniform_(self.convolutions[ii][0].weight)
            torch.nn.init.xavier_uniform_(self.convolutions[ii][2].weight)
            torch.nn.init.xavier_uniform_(self.convolutions[ii][4].weight)
            torch.nn.init.xavier_uniform_(self.convolutions[ii][6].weight)
            torch.nn.init.xavier_uniform_(self.convolutions[ii][8].weight)
            torch.nn.init.xavier_uniform_(self.convolutions[ii][10].weight)
            torch.nn.init.xavier_uniform_(self.convolutions[ii][12].weight)
            torch.nn.init.xavier_uniform_(self.convolutions[ii][14].weight)

        self.softmax_layer = nn.Linear(256, 256)

        torch.nn.init.xavier_uniform_(self.softmax_layer.weight)

        self.act = nn.Softmax(dim=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, mgc, signal=None, prev=None, training=False):
        # x = x.reshape(x.shape[0], 1, x.shape[1])

        if signal is not None:
            # prepare the input
            cond = self.conditioning(torch.Tensor(mgc).to(device))

            x_list = []
            for ii in range(len(signal) - self.RECEPTIVE_FIELD):
                x_list.append(signal[ii:ii + self.RECEPTIVE_FIELD])

            x = torch.Tensor(x_list).to(device)
            x = x.reshape(x.shape[0], 1, x.shape[1])
            pre_softmax = []

            for ii in range(self.NUM_NETWORKS):
                pre_softmax.append(self.convolutions[ii](x))

            pre = cond
            for ii in range(self.NUM_NETWORKS):
                pre = pre + pre_softmax[ii].reshape(cond.shape[0], cond.shape[1])

            softmax = self.softmax_layer(pre)  # self.act()
            # from ipdb import set_trace
            # set_trace()
        else:
            signal = prev[-self.RECEPTIVE_FIELD:]
            for ii in range(len(mgc)):
                x = torch.Tensor(signal[-self.RECEPTIVE_FIELD:]).to(device)
                x = x.reshape(1, 1, x.shape[0])
                cond = self.conditioning(torch.Tensor(mgc[ii]).to(device).reshape(1, 60))

                pre_softmax = []

                for ii in range(self.NUM_NETWORKS):
                    pre_softmax.append(self.convolutions[ii](x))

                pre = pre_softmax[0]
                for ii in range(1, self.NUM_NETWORKS):
                    pre = pre + pre_softmax[ii].reshape(cond.shape[0], cond.shape[1])

                pre = pre * cond

                # pre = pre.reshape(pre.shape[0 ], pre.shape[1])
                softmax = self.act(self.softmax_layer(pre))
                sample = self._pick_sample(softmax.data.cpu().numpy().reshape(256), temperature=0.8)
                f = float(sample) / 128 - 1.0
                sign = np.sign(f)
                decoded = sign * (1.0 / 255.0) * (pow(1.0 + 255, abs(f)) - 1.0)
                signal.append(decoded)

        return signal[self.RECEPTIVE_FIELD:], softmax

    def _pick_sample(self, probs, temperature=1.0):
        probs = probs / np.sum(probs)
        scaled_prediction = np.log(probs) / temperature
        scaled_prediction = (scaled_prediction -
                             np.logaddexp.reduce(scaled_prediction))
        scaled_prediction = np.exp(scaled_prediction)
        # print np.sum(probs)
        # probs = probs / np.sum(probs)
        return np.random.choice(np.arange(256), p=scaled_prediction)
