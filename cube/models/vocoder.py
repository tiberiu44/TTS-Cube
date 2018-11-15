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
        self.abs_loss = torch.nn.L1Loss()
        self.bce_loss = torch.nn.BCELoss()
        self.cnt = 0

    def synthesize(self, mgc, batch_size, sample=True, temperature=1.0, path=None):
        last_proc = 0
        synth = []
        x = []
        for mgc_index in range(len(mgc)):
            curr_proc = int((mgc_index + 1) * 100 / len(mgc))
            if curr_proc % 5 == 0 and curr_proc != last_proc:
                while last_proc < curr_proc:
                    last_proc += 5
                    sys.stdout.write(' ' + str(last_proc))
                    sys.stdout.flush()

            input = mgc[mgc_index]

            x.append(input)

            if len(x) == batch_size:
                inp = torch.tensor(x).reshape(batch_size, 60).float().to(device)
                [output, mean, logvar, mean_student, logvar_student] = self.network(inp)
                output = output.reshape(self.UPSAMPLE_COUNT * batch_size)
                for zz in output:
                    synth.append(zz.item())
                x = []

        if len(x) != 0:
            inp = torch.tensor(x).reshape(len(x), 60).float().to(device)
            [output, mean, logvar, mean_student, logvar_student] = self.network(inp)
            output = output.reshape(self.UPSAMPLE_COUNT * len(x))
            for x in output:
                synth.append(x.item())

        # synth = self.dio.ulaw_decode(synth, discreete=False)
        synth = np.array(synth, dtype=np.float32)
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

    def _get_loss(self, signal_orig, signal_pred, mean, logvar, mean_student, logvar_student, batch_size):
        if batch_size < 4:
            return None
        fft_orig = torch.stft(signal_orig.reshape(batch_size * self.UPSAMPLE_COUNT), n_fft=512,
                              window=torch.hann_window(window_length=512).to(device))
        fft_pred = torch.stft(signal_pred.reshape(batch_size * self.UPSAMPLE_COUNT), n_fft=512,
                              window=torch.hann_window(window_length=512).to(device))
        loss = self.abs_loss(fft_pred, fft_orig)  # torch.abs(torch.abs(fft_orig) - torch.abs(fft_pred)).sum() / (
        # fft_orig.shape[0] * fft_orig.shape[1] * fft_orig.shape[2])
        # from ipdb import set_trace
        # set_trace()

        power_orig = fft_orig * fft_orig
        power_pred = fft_pred * fft_pred
        power_orig = torch.sum(power_orig, dim=2)
        power_pred = torch.sum(power_pred, dim=2)
        loss += self.abs_loss(power_pred, power_orig)
        # from ipdb import set_trace
        # set_trace()
        mean = mean.reshape(mean.shape[0], mean.shape[1])
        logvar = logvar.reshape(mean.shape[0], mean.shape[1])
        loss += self.abs_loss(mean_student, torch.tensor(mean, requires_grad=False))
        loss += self.abs_loss(logvar_student, torch.tensor(logvar, requires_grad=False))

        # loss += self.abs_loss(torch.log(power_pred + 1e-5), torch.log(power_orig + 1e-5))

        # loss += self.bce_loss(signal_pred, signal_orig)

        loss += -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        return loss

    def learn(self, wave, mgc, batch_size):
        last_proc = 0
        total_loss = 0
        losses = []
        cnt = 0
        noise = np.random.normal(0, 1.0, (len(wave) + self.UPSAMPLE_COUNT))
        x = []
        y = []
        num_batches = 0
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
                batch_x = torch.tensor(x).reshape(batch_size, 60).float().to(device)
                batch_y = torch.tensor(y).reshape(batch_size, self.UPSAMPLE_COUNT).to(device)
                [y_pred, mean, logvar, mean_student, logvar_student] = self.network(batch_x, signal=batch_y,
                                                                                    training=True)

                loss = self._get_loss(batch_y, y_pred, mean, logvar, mean_student, logvar_student, batch_size)
                loss.backward()
                self.trainer.step()
                total_loss += loss
                x = []
                y = []
                num_batches += 1

        if len(x) > 0:
            self.trainer.zero_grad()
            batch_x = torch.tensor(x).reshape(len(x), 60).float().to(device)
            batch_y = torch.tensor(y).reshape(len(x), self.UPSAMPLE_COUNT).to(device)
            [y_pred, mean, logvar, mean_student, logvar_student] = self.network(batch_x, signal=batch_y, training=True)

            loss = self._get_loss(batch_y, y_pred, mean, logvar, mean_student, logvar_student, len(x))
            if loss is not None:
                loss.backward()
                self.trainer.step()
                total_loss += loss
                num_batches += 1

        total_loss = total_loss.item()
        # self.cnt += 1
        return total_loss / num_batches


class VocoderNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(VocoderNetwork, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 200, kernel_size=16, stride=1, padding=0),
            # nn.ELU()
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
            nn.Conv1d(1, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 200, kernel_size=16, stride=1, padding=0),

            # nn.ELU()

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
            nn.Conv1d(1, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 200, kernel_size=16, stride=1, padding=0),

            # nn.ELU()

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
            nn.Conv1d(1, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 200, kernel_size=16, stride=1, padding=0),

            # nn.ELU()

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
            nn.Conv1d(1, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 200, kernel_size=16, stride=1, padding=0),

            # nn.ELU()

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
            nn.Conv1d(1, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 200, kernel_size=16, stride=1, padding=0),

            # nn.ELU()

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
            nn.Conv1d(1, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 200, kernel_size=16, stride=1, padding=0),

            # nn.ELU()

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
            nn.Conv1d(1, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=13, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 200, kernel_size=16, stride=1, padding=0),

            # nn.ELU()

        )

        torch.nn.init.xavier_uniform_(self.net8[0].weight)
        torch.nn.init.xavier_uniform_(self.net8[2].weight)
        torch.nn.init.xavier_uniform_(self.net8[4].weight)
        torch.nn.init.xavier_uniform_(self.net8[6].weight)
        torch.nn.init.xavier_uniform_(self.net8[8].weight)
        torch.nn.init.xavier_uniform_(self.net8[10].weight)
        torch.nn.init.xavier_uniform_(self.net8[12].weight)
        torch.nn.init.xavier_uniform_(self.net8[14].weight)

        self.encoder_signal_mean = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 60, kernel_size=8, stride=1, padding=0),
        )

        torch.nn.init.xavier_uniform_(self.encoder_signal_mean[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_mean[2].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_mean[4].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_mean[6].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_mean[8].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_mean[10].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_mean[12].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_mean[14].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_mean[16].weight)

        self.encoder_signal_logvar = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 256, kernel_size=25, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(256, 60, kernel_size=8, stride=1, padding=0),
        )

        torch.nn.init.xavier_uniform_(self.encoder_signal_logvar[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_logvar[2].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_logvar[4].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_logvar[6].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_logvar[8].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_logvar[10].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_logvar[12].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_logvar[14].weight)
        torch.nn.init.xavier_uniform_(self.encoder_signal_logvar[16].weight)

        self.encoder_mean = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.Tanh(),
            nn.Linear(400, 200),
            nn.Tanh(),
            nn.Linear(200, 60)
        )
        # torch.nn.init.xavier_uniform_(self.encoder_mean[0].weight)
        # torch.nn.init.xavier_uniform_(self.encoder_mean[2].weight)
        # torch.nn.init.xavier_uniform_(self.encoder_mean[4].weight)

        self.encoder_logvar = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.Tanh(),
            nn.Linear(400, 200),
            nn.Tanh(),
            nn.Linear(200, 60)
        )
        # torch.nn.init.xavier_uniform_(self.encoder_logvar[0].weight)
        # torch.nn.init.xavier_uniform_(self.encoder_logvar[2].weight)
        # torch.nn.init.xavier_uniform_(self.encoder_logvar[4].weight)

        self.act = nn.Softsign()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, signal=None, training=False):
        # if training:
        #    x = torch.nn.functional.dropout(x, p=0.33, training=True)

        # from ipdb import set_trace
        # set_trace()

        if signal is not None:
            mean = self.encoder_signal_mean(signal.reshape(signal.shape[0], 1, signal.shape[1]))
            logvar = self.encoder_signal_mean(signal.reshape(signal.shape[0], 1, signal.shape[1]))
            mean_student = self.encoder_mean(x)
            logvar_student = self.encoder_logvar(x)
        else:
            mean = self.encoder_mean(x)
            logvar = self.encoder_logvar(x)
            mean_student = None
            logvar_student = None

        x = self.reparameterize(mean, logvar).reshape(mean.shape[0], 1, mean.shape[1])

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

        return [self.act(out1 + out2 + out3 + out4 + out5 + out6 + out7 + out8), mean, logvar, mean_student,
                logvar_student]
