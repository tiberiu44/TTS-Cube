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
        self.mse_loss = torch.nn.MSELoss()
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
                [output, mean, logvar] = self.network(inp)
                output = output.reshape(self.UPSAMPLE_COUNT * batch_size)
                for zz in output:
                    synth.append(zz.item())
                x = []

        if len(x) != 0:
            inp = torch.tensor(x).reshape(len(x), 60).float().to(device)
            [output, mean, logvar] = self.network(inp)
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

    def _get_loss(self, signal_orig, signal_pred, p_coarse, p_fine, batch_size):

        # loss = 0
        target_fine = []
        target_coarse = []
        for y_batch in signal_orig:
            coarse = []
            fine = []
            for yy in y_batch:
                b16 = int((yy + 1.0) / 2.0 * 65535)
                coarse.append((b16 // 256) / 255)
                fine.append((b16 % 256) / 255)
            target_fine.append(fine)
            target_coarse.append(coarse)
        # from ipdb import set_trace
        # set_trace()
        t_fine = torch.tensor(target_fine).to(device)
        t_coarse = torch.tensor(target_coarse).to(device)
        loss = self.mse_loss(p_coarse.reshape(p_coarse.shape[0], p_coarse.shape[1]), t_coarse) + self.mse_loss(
            p_fine.reshape(p_coarse.shape[0], p_coarse.shape[1]), t_fine)

        # if batch_size < 4:
        #     return None
        # fft_orig = torch.stft(signal_orig.reshape(batch_size * self.UPSAMPLE_COUNT), n_fft=512,
        #                       window=torch.hann_window(window_length=512).to(device))
        # fft_pred = torch.stft(signal_pred.reshape(batch_size * self.UPSAMPLE_COUNT), n_fft=512,
        #                       window=torch.hann_window(window_length=512).to(device))
        # loss = self.mse_loss(fft_pred, fft_orig)  # torch.abs(torch.abs(fft_orig) - torch.abs(fft_pred)).sum() / (
        # # fft_orig.shape[0] * fft_orig.shape[1] * fft_orig.shape[2])
        # # from ipdb import set_trace
        # # set_trace()
        #
        # power_orig = fft_orig * fft_orig
        # power_pred = fft_pred * fft_pred
        # power_orig = torch.sum(power_orig, dim=2)
        # power_pred = torch.sum(power_pred, dim=2)
        # loss += self.abs_loss(power_pred, power_orig)
        # # from ipdb import set_trace
        # # set_trace()
        # mean = mean.reshape(mean.shape[0], mean.shape[1])
        # logvar = logvar.reshape(mean.shape[0], mean.shape[1])
        #
        # # loss += self.abs_loss(torch.log(power_pred + 1e-5), torch.log(power_orig + 1e-5))
        #
        # # loss += self.mse_loss(signal_pred.reshape(signal_pred.shape[0], signal_pred.shape[2]), signal_orig) * 7
        #
        # loss += (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())) / (batch_size * self.UPSAMPLE_COUNT)
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
                # batch_y = torch.tensor(y).reshape(batch_size, self.UPSAMPLE_COUNT).to(device)
                [y_pred, coarse, fine] = self.network(batch_x,
                                                      training=True)

                loss = self._get_loss(y, y_pred, coarse, fine, batch_size)
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
            [y_pred, coarse, fine] = self.network(batch_x, training=True)

            loss = self._get_loss(batch_y, y_pred, coarse, fine, len(x))
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

        self.conv_net_coarse = nn.ModuleList([nn.Sequential(
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
            nn.Conv1d(256, 200, kernel_size=16, stride=1, padding=0)) for ii in range(8)]
        )
        self.conv_net_fine = nn.ModuleList([nn.Sequential(
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
            nn.Conv1d(256, 200, kernel_size=16, stride=1, padding=0)) for ii in range(8)]
        )

        for ii in range(8):
            torch.nn.init.xavier_uniform_(self.conv_net_coarse[ii][0].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_coarse[ii][2].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_coarse[ii][4].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_coarse[ii][6].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_coarse[ii][8].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_coarse[ii][10].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_coarse[ii][12].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_coarse[ii][14].weight)

            torch.nn.init.xavier_uniform_(self.conv_net_fine[ii][0].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_fine[ii][2].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_fine[ii][4].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_fine[ii][6].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_fine[ii][8].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_fine[ii][10].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_fine[ii][12].weight)
            torch.nn.init.xavier_uniform_(self.conv_net_fine[ii][14].weight)

        self.act = nn.Sigmoid()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, signal=None, training=False):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        out_coarse = []
        out_fine = []
        for ii in range(8):
            out_coarse.append(self.conv_net_coarse[ii](x))
            out_fine.append(self.conv_net_fine[ii](x))

        coarse = out_coarse[0] + out_coarse[1] + out_coarse[2] + out_coarse[3] + out_coarse[4] + out_coarse[5] + \
                 out_coarse[6] + out_coarse[7]
        fine = out_fine[0] + out_fine[1] + out_fine[2] + out_fine[3] + out_fine[4] + out_fine[5] + \
               out_fine[6] + out_fine[7]
        # from ipdb import set_trace
        # set_trace()
        # fine = self.act(fine)
        # coarse = self.act(coarse)

        # x = self.reparameterize(mean, logvar).reshape(mean.shape[0], 1, mean.shape[1])
        x = ((coarse + fine / 256) - 0.5) * 2
        return x, coarse, fine
