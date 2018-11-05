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
        self.HIDDEN_SIZE = [1000, 1000]

        self.FFT_SIZE = 513
        self.UPSAMPLE_COUNT = int(12.5 * params.target_sample_rate / 1000)
        self.FILTER_SIZE = 128

        self.sparse = False
        self.dio = DatasetIO()
        self.vocoder = MelVocoder()

        self.network = VocoderNetwork(self.params.mgc_order, self.UPSAMPLE_COUNT).to(device)
        self.trainer = torch.optim.Adam(self.network.parameters(), lr=self.params.learning_rate)

    def synthesize(self, mgc, batch_size, sample=True, temperature=1.0, path=None):
        last_proc = 0
        synth = []
        noise = np.random.normal(0, 1.0, (len(mgc) * self.UPSAMPLE_COUNT + self.UPSAMPLE_COUNT))
        for mgc_index in range(len(mgc)):
            curr_proc = int((mgc_index + 1) * 100 / len(mgc))
            if curr_proc % 5 == 0 and curr_proc != last_proc:
                while last_proc < curr_proc:
                    last_proc += 5
                    sys.stdout.write(' ' + str(last_proc))
                    sys.stdout.flush()

            # output, excitation, filter, vuv = self._predict_one(mgc[mgc_index],
            #                                                     noise[
            #                                                     self.UPSAMPLE_COUNT * mgc_index:self.UPSAMPLE_COUNT * mgc_index + 2 * self.UPSAMPLE_COUNT])
            input = mgc[mgc_index]
            inp = torch.tensor(input).reshape(1, 1, 60).float().to(device)
            output = self.network(inp).reshape(self.UPSAMPLE_COUNT)
            for x in output:
                synth.append(x.item())

        # synth = self.dio.ulaw_decode(synth, discreete=False)
        synth = np.array(synth, dtype=np.float32)
        synth = np.clip(synth * 32768, -32767, 32767)
        synth = np.array(synth, dtype=np.int16)

        return synth

    def store(self, output_base):
        # self.model.save(output_base + ".network")
        x = 0

    def load(self, output_base):
        self.model.populate(output_base + ".network")

    def _predict_one(self, mgc, noise):

        return None

    def learn(self, wave, mgc, batch_size):
        # from ipdb import set_trace
        # set_trace()
        # disc, wave = self.dio.ulaw_encode(wave)
        last_proc = 0
        total_loss = 0
        losses = []
        cnt = 0
        noise = np.random.normal(0, 1.0, (len(wave) + self.UPSAMPLE_COUNT))
        x = []
        y = []
        for mgc_index in range(len(mgc)):
            curr_proc = int((mgc_index + 1) * 100 / len(mgc))
            if curr_proc % 5 == 0 and curr_proc != last_proc:
                while last_proc < curr_proc:
                    last_proc += 5
                    sys.stdout.write(' ' + str(last_proc))
                    sys.stdout.flush()

            if mgc_index < len(mgc) - 1:
                cnt += 1
                input = mgc[mgc_index]
                output = wave[self.UPSAMPLE_COUNT * mgc_index:self.UPSAMPLE_COUNT * mgc_index + self.UPSAMPLE_COUNT]
                x.append(input)
                y.append(output)

            if len(x) == batch_size:
                self.trainer.zero_grad()
                batch_x = torch.tensor(x).reshape(batch_size, 1, 60).float().to(device)
                batch_y = torch.tensor(y).reshape(batch_size, self.UPSAMPLE_COUNT).to(device)
                y_pred = self.network(batch_x)
                # loss = self.criterion.forward(y_pred, batch_y.to(device))
                loss = (y_pred - batch_y).pow(2).sum() / len(x)
                loss.backward()
                self.trainer.step()
                total_loss += loss
                x = []
                y = []

        if len(x) > 0:
            self.trainer.zero_grad()
            batch_x = torch.tensor(x).reshape(len(x), 1, 60).float().to(device)
            batch_y = torch.tensor(y).reshape(len(x), self.UPSAMPLE_COUNT).to(device)
            y_pred = self.network(batch_x)
            # loss = self.criterion.forward(y_pred, batch_y.to(device))
            loss = (y_pred - batch_y).pow(2).sum() / len(x)
            loss.backward()
            self.trainer.step()
            total_loss += loss

        total_loss = total_loss.item()
        return total_loss / cnt


class VocoderNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(VocoderNetwork, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(2, 4, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(4, 8, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(128, 200, kernel_size=16, stride=1, padding=0),
            # nn.Tanh()
        )
        torch.nn.init.xavier_uniform_(self.net1[0].weight)
        torch.nn.init.xavier_uniform_(self.net1[2].weight)
        torch.nn.init.xavier_uniform_(self.net1[4].weight)
        torch.nn.init.xavier_uniform_(self.net1[6].weight)
        torch.nn.init.xavier_uniform_(self.net1[8].weight)
        torch.nn.init.xavier_uniform_(self.net1[10].weight)
        torch.nn.init.xavier_uniform_(self.net1[12].weight)
        torch.nn.init.xavier_uniform_(self.net1[14].weight)

        self.net2 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(2, 4, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(4, 8, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(128, 200, kernel_size=16, stride=1, padding=0),
            # nn.Tanh()
        )
        torch.nn.init.xavier_uniform_(self.net2[0].weight)
        torch.nn.init.xavier_uniform_(self.net2[2].weight)
        torch.nn.init.xavier_uniform_(self.net2[4].weight)
        torch.nn.init.xavier_uniform_(self.net2[6].weight)
        torch.nn.init.xavier_uniform_(self.net2[8].weight)
        torch.nn.init.xavier_uniform_(self.net2[10].weight)
        torch.nn.init.xavier_uniform_(self.net2[12].weight)
        torch.nn.init.xavier_uniform_(self.net2[14].weight)

        self.net3 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(2, 4, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(4, 8, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(128, 200, kernel_size=16, stride=1, padding=0),
            # nn.Tanh()
        )
        torch.nn.init.xavier_uniform_(self.net3[0].weight)
        torch.nn.init.xavier_uniform_(self.net3[2].weight)
        torch.nn.init.xavier_uniform_(self.net3[4].weight)
        torch.nn.init.xavier_uniform_(self.net3[6].weight)
        torch.nn.init.xavier_uniform_(self.net3[8].weight)
        torch.nn.init.xavier_uniform_(self.net3[10].weight)
        torch.nn.init.xavier_uniform_(self.net3[12].weight)
        torch.nn.init.xavier_uniform_(self.net3[14].weight)

        self.net4 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(2, 4, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(4, 8, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(128, 200, kernel_size=16, stride=1, padding=0),
            # nn.Tanh()
        )
        torch.nn.init.xavier_uniform_(self.net4[0].weight)
        torch.nn.init.xavier_uniform_(self.net4[2].weight)
        torch.nn.init.xavier_uniform_(self.net4[4].weight)
        torch.nn.init.xavier_uniform_(self.net4[6].weight)
        torch.nn.init.xavier_uniform_(self.net4[8].weight)
        torch.nn.init.xavier_uniform_(self.net4[10].weight)
        torch.nn.init.xavier_uniform_(self.net4[12].weight)
        torch.nn.init.xavier_uniform_(self.net4[14].weight)

        self.net5 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(2, 4, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(4, 8, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(128, 200, kernel_size=16, stride=1, padding=0),
            # nn.Tanh()
        )
        torch.nn.init.xavier_uniform_(self.net5[0].weight)
        torch.nn.init.xavier_uniform_(self.net5[2].weight)
        torch.nn.init.xavier_uniform_(self.net5[4].weight)
        torch.nn.init.xavier_uniform_(self.net5[6].weight)
        torch.nn.init.xavier_uniform_(self.net5[8].weight)
        torch.nn.init.xavier_uniform_(self.net5[10].weight)
        torch.nn.init.xavier_uniform_(self.net5[12].weight)
        torch.nn.init.xavier_uniform_(self.net5[14].weight)

        self.net6 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(2, 4, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(4, 8, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(128, 200, kernel_size=16, stride=1, padding=0),
            # nn.Tanh()
        )
        torch.nn.init.xavier_uniform_(self.net6[0].weight)
        torch.nn.init.xavier_uniform_(self.net6[2].weight)
        torch.nn.init.xavier_uniform_(self.net6[4].weight)
        torch.nn.init.xavier_uniform_(self.net6[6].weight)
        torch.nn.init.xavier_uniform_(self.net6[8].weight)
        torch.nn.init.xavier_uniform_(self.net6[10].weight)
        torch.nn.init.xavier_uniform_(self.net6[12].weight)
        torch.nn.init.xavier_uniform_(self.net6[14].weight)

        self.net7 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(2, 4, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(4, 8, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(128, 200, kernel_size=16, stride=1, padding=0),
            # nn.Tanh()
        )
        torch.nn.init.xavier_uniform_(self.net7[0].weight)
        torch.nn.init.xavier_uniform_(self.net7[2].weight)
        torch.nn.init.xavier_uniform_(self.net7[4].weight)
        torch.nn.init.xavier_uniform_(self.net7[6].weight)
        torch.nn.init.xavier_uniform_(self.net7[8].weight)
        torch.nn.init.xavier_uniform_(self.net7[10].weight)
        torch.nn.init.xavier_uniform_(self.net7[12].weight)
        torch.nn.init.xavier_uniform_(self.net7[14].weight)

        self.net8 = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(2, 4, kernel_size=13, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(4, 8, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv1d(128, 200, kernel_size=16, stride=1, padding=0),
            # nn.Tanh()
        )
        torch.nn.init.xavier_uniform_(self.net8[0].weight)
        torch.nn.init.xavier_uniform_(self.net8[2].weight)
        torch.nn.init.xavier_uniform_(self.net8[4].weight)
        torch.nn.init.xavier_uniform_(self.net8[6].weight)
        torch.nn.init.xavier_uniform_(self.net8[8].weight)
        torch.nn.init.xavier_uniform_(self.net8[10].weight)
        torch.nn.init.xavier_uniform_(self.net8[12].weight)
        torch.nn.init.xavier_uniform_(self.net8[14].weight)

        self.act = nn.Tanh()

    def forward(self, x):
        out1 = self.net1(x)
        out1 = out1.reshape(out1.size(0), out1.size(1) * out1.size(2))

        out2 = self.net2(x)
        out2 = out2.reshape(out2.size(0), out2.size(1) * out2.size(2))

        out3 = self.net3(x)
        out3 = out3.reshape(out3.size(0), out3.size(1) * out3.size(2))

        out4 = self.net4(x)
        out4 = out4.reshape(out4.size(0), out4.size(1) * out4.size(2))

        out5 = self.net5(x)
        out5 = out5.reshape(out5.size(0), out5.size(1) * out5.size(2))

        out6 = self.net6(x)
        out6 = out6.reshape(out6.size(0), out6.size(1) * out6.size(2))

        out7 = self.net7(x)
        out7 = out7.reshape(out7.size(0), out7.size(1) * out7.size(2))

        out8 = self.net8(x)
        out8 = out8.reshape(out8.size(0), out8.size(1) * out8.size(2))

        return self.act(out1 + out2 + out3 + out4 + out5 + out6 + out7 + out8)
