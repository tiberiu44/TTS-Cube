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
import tqdm
import numpy as np
from models.clarinet.wavenet import Wavenet
from models.clarinet.modules import GaussianLoss, STFT, KL_Loss
from models.clarinet.wavenet_iaf import Wavenet_Student
from torch.distributions.normal import Normal

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _create_batches(y_target, mgc, batch_size, UPSAMPLE_COUNT=256, mgc_order=60):
    x_list = []
    y_list = []
    c_list = []

    x_mini_list = []
    y_mini_list = []
    c_mini_list = []
    mini_batch = 25

    for batch_index in range((len(mgc) - 1) // mini_batch):
        mgc_start = batch_index * mini_batch
        mgc_stop = batch_index * mini_batch + mini_batch
        c_mini_list.append(mgc[mgc_start:mgc_stop].reshape(mini_batch, mgc_order).transpose())
        x_start = batch_index * mini_batch * UPSAMPLE_COUNT
        x_stop = batch_index * mini_batch * UPSAMPLE_COUNT + mini_batch * UPSAMPLE_COUNT
        x_mini_list.append(y_target[x_start:x_stop].reshape(1, x_stop - x_start))
        y_mini_list.append(y_target[x_start:x_stop].reshape(x_stop - x_start, 1))

        if len(c_mini_list) == batch_size:
            x_list.append(torch.tensor(x_mini_list).to(device))
            y_list.append(torch.tensor(y_mini_list).to(device))
            c_list.append(torch.tensor(c_mini_list).to(device))
            c_mini_list = []
            x_mini_list = []
            y_mini_list = []

    if len(c_mini_list) != 0:
        x_list.append(torch.tensor(x_mini_list).to(device))
        y_list.append(torch.tensor(y_mini_list).to(device))
        c_list.append(torch.tensor(c_mini_list).to(device))
    # from ipdb import set_trace
    # set_trace()

    return x_list, y_list, c_list


class Vocoder:
    def __init__(self, params):

        self.params = params

        self.UPSAMPLE_COUNT = int(16 * params.target_sample_rate / 1000)
        self.RECEPTIVE_SIZE = 3 * 3 * 3 * 3 * 3 * 3
        self.model = Wavenet(out_channels=2,
                             num_blocks=4,
                             num_layers=6,
                             residual_channels=128,
                             gate_channels=256,
                             skip_channels=128,
                             kernel_size=3,
                             cin_channels=params.mgc_order,
                             upsample_scales=[16, 16]).to(device)

        self.loss = GaussianLoss()

        self.trainer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate)

    def learn(self, y_target, mgc, batch_size):
        # prepare batches
        x_list, y_list, c_list = _create_batches(y_target, mgc, batch_size, UPSAMPLE_COUNT=self.UPSAMPLE_COUNT,
                                                 mgc_order=self.params.mgc_order)
        if len(x_list) == 0:
            return 0
        # learn
        total_loss = 0
        for x, y, c in tqdm.tqdm(zip(x_list, y_list, c_list), total=len(c_list)):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            c = torch.tensor(c, dtype=torch.float32).to(device)
            self.trainer.zero_grad()
            y_hat = self.model(x, c)

            t_y = y[:, 1:]  # .reshape(1, y_hat.shape[0] * y_hat.shape[2] - 1, 1)
            p_y = y_hat[:, :, :-1]

            loss = self.loss(p_y, t_y, size_average=True)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
            self.trainer.step()
        return total_loss / len(x_list)

    def synthesize(self, mgc, batch_size):
        num_samples = len(mgc) * self.UPSAMPLE_COUNT
        # from ipdb import set_trace
        # set_trace()
        with torch.no_grad():
            c = torch.tensor(mgc.transpose(), dtype=torch.float32).to(device).reshape(1, mgc[0].shape[0], len(mgc))
            x = self.model.generate(num_samples - 1, c, device=device)
        torch.cuda.synchronize()
        x = x.squeeze().numpy() * 32768
        return x

    def store(self, output_base):
        torch.save(self.model.state_dict(), output_base + ".network")

    def load(self, output_base):
        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1:
                self.model.load_state_dict(torch.load(output_base + ".network", map_location='cuda:0'))
            else:
                self.model.load_state_dict(torch.load(output_base + ".network"))
        else:
            self.model.load_state_dict(
                torch.load(output_base + '.network', map_location=lambda storage, loc: storage))
        self.model.to(device)


class ParallelVocoder:
    def __init__(self, params, vocoder=None):
        self.UPSAMPLE_COUNT = int(16 * params.target_sample_rate / 1000)
        self.RECEPTIVE_SIZE = 3 * 3 * 3 * 3 * 3 * 3
        self.params = params
        self.model_t = vocoder.model
        self.model_s = Wavenet_Student(num_blocks_student=[1, 1, 1, 4],
                                       num_layers=6, cin_channels=self.params.mgc_order)
        self.model_s.to(device)

        self.stft = STFT(filter_length=1024, hop_length=256).to(device)
        self.criterion_t = KL_Loss().to(device)
        self.criterion_frame = torch.nn.MSELoss().to(device)
        self.trainer = torch.optim.Adam(self.model_s.parameters(), lr=self.params.learning_rate)
        self.model_t.eval()
        self.model_s.train()

    def learn(self, y_target, mgc, batch_size):
        # prepare batches
        x_list, y_list, c_list = _create_batches(y_target, mgc, batch_size, UPSAMPLE_COUNT=self.UPSAMPLE_COUNT,
                                                 mgc_order=self.params.mgc_order)
        if len(x_list) == 0:
            return 0
        # learn
        total_loss = 0
        for x, y, c in tqdm.tqdm(zip(x_list, y_list, c_list), total=len(c_list)):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            c = torch.tensor(c, dtype=torch.float32).to(device)
            self.trainer.zero_grad()
            q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
            z = q_0.sample()
            c_up = self.model_t.upsample(c)
            x_student, mu_s, logs_s = self.model_s(z, c_up)
            with torch.no_grad():
                mu_logs_t = self.model_t(x_student, c)
            mu_logs_t = mu_logs_t.detach()

            loss_t, loss_KL, loss_reg = self.criterion_t(mu_s, logs_s, mu_logs_t[:, 0:1, :-1], mu_logs_t[:, 1:, :-1])
            stft_student, _ = self.stft(x_student[:, :, 1:])
            stft_truth, _ = self.stft(x[:, :, 1:])
            loss_frame = self.criterion_frame(stft_student, stft_truth)
            loss_tot = loss_t + loss_frame
            total_loss += loss_tot.item()
            loss_tot.backward()

            torch.nn.utils.clip_grad_norm_(self.model_s.parameters(), 10.)
            self.trainer.step()

        return total_loss / len(x_list)

    def synthesize(self, mgc, batch_size):
        num_samples = len(mgc) * self.UPSAMPLE_COUNT
        zeros = np.zeros((1, 1, num_samples))
        ones = np.ones((1, 1, num_samples))
        with torch.no_grad():
            c = torch.tensor(mgc.transpose(), dtype=torch.float32).to(device).reshape(1, mgc[0].shape[0], len(mgc))
            c_up = self.model_t.upsample(c)
            q_0 = Normal(torch.tensor(zeros, dtype=torch.float32).to(device),
                         torch.tensor(ones, dtype=torch.float32).to(device))
            z = q_0.sample()
            x = self.model_s.generate(z, c_up, device=device)
        torch.cuda.synchronize()
        x = x.squeeze().cpu().numpy() * 32768
        return x

    def store(self, output_base):
        torch.save(self.model_s.state_dict(), output_base + ".network")

    def load(self, output_base):
        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1:
                self.model_s.load_state_dict(torch.load(output_base + ".network", map_location='cuda:0'))
            else:
                self.model_s.load_state_dict(torch.load(output_base + ".network"))
        else:
            self.model_s.load_state_dict(
                torch.load(output_base + '.network', map_location=lambda storage, loc: storage))
        self.model_s.to(device)
