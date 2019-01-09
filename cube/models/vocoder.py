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
from models.clarinet.modules import GaussianLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Vocoder:
    def __init__(self, params):

        self.params = params

        self.UPSAMPLE_COUNT = int(12.5 * params.target_sample_rate / 1000)
        self.RECEPTIVE_SIZE = 3 * 3 * 3 * 3 * 3 * 3
        self.model = Wavenet(out_channels=2,
                             num_blocks=4,
                             num_layers=6,
                             residual_channels=128,
                             gate_channels=256,
                             skip_channels=128,
                             kernel_size=3,
                             cin_channels=60,
                             upsample_scales=[10, 20]).to(device)

        self.loss = GaussianLoss()

        self.trainer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate)

    def _create_batches(self, y_target, mgc, batch_size):
        x_list = []
        y_list = []
        c_list = []

        for mgc_index in range(len(mgc)):
            c_list.append(mgc[mgc_index])

            y_start = mgc_index * self.UPSAMPLE_COUNT
            y_stop = mgc_index * self.UPSAMPLE_COUNT + self.UPSAMPLE_COUNT
            y_list.append(y_target[y_start:y_stop])

            x_stop = y_stop - 1
            x_start = y_start - self.RECEPTIVE_SIZE
            if x_start < 0:
                x_start = 0
            x_tmp = y_target[x_start:x_stop]

            while x_tmp.shape[0] < self.UPSAMPLE_COUNT + self.RECEPTIVE_SIZE - 1:
                x_tmp = np.insert(x_tmp, 0, 0)
            x_list.append(x_tmp)

        return x_list, y_list, c_list

    def learn(self, y_target, mgc, batch_size):
        # prepare batches
        x_list, y_list, c_list = self._create_batches(y_target, mgc, batch_size)
        # learn
        total_loss = 0
        for x, y, c in tqdm.tqdm(zip(x_list, y_list, c_list)):
            x, y, c = torch.tensor(x).to(device), torch.tensor(y).to(device), torch.tensor(c).to(device).reshape(1, 1,
                                                                                                                 60)
            self.trainer.zero_grad()
            y_hat = self.model(x, c)
            loss = self.loss(y_hat[:, :, :-1], y[:, 1:, :], size_average=True)
            total_loss += loss.item() / len(y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
            self.trainer.step()
        return total_loss / len(x_list)

    def synthesize(self, mgc, batch_size):
        pass

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
    def __init__(self, wavenet=None):
        pass

    def learn(self):
        pass

    def synthesize(self):
        pass

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass
