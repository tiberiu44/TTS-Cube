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


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.to(device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


class BeeCoder:
    def __init__(self, params, model=None, runtime=False):
        self.params = params

        self.UPSAMPLE_COUNT = int(12.5 * params.target_sample_rate / 1000)
        self.RECEPTIVE_SIZE = 512  # this means 32ms

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

            # x = [input for ii in range(self.UPSAMPLE_COUNT)]

            [signal, means, logvars, logits, aux] = self.network([input], prev=synth)
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

    def _compute_mixture_loss(self, y_target, means, logvars, logit_probs):
        num_classes = 65536
        # from ipdb import set_trace
        # set_trace()

        y = y_target.reshape(y_target.shape[0], 1).expand_as(means)

        centered_y = y - means
        inv_stdv = torch.exp(-logvars)
        plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - torch.nn.functional.softplus(plus_in)
        log_one_minus_cdf_min = -torch.nn.functional.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered_y
        log_pdf_mid = mid_in - logvars - 2. * torch.nn.functional.softplus(mid_in)
        inner_inner_cond = (cdf_delta > 1e-5).float()
        inner_inner_out = inner_inner_cond * \
                          torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
                          (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
        inner_cond = (y > 0.999).float()
        inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
        cond = (y < -0.999).float()
        log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

        log_probs = log_probs + torch.nn.functional.log_softmax(logit_probs, -1)
        return -torch.sum(log_sum_exp(log_probs)) / y_target.shape[0]

    def _compute_aux_loss(self, y_target, y_pred):

        signal_target = y_target  # y_target.reshape(y_target.shape[0] * y_target.shape[1])
        signal_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1])
        fft_target = torch.stft(signal_target, 512, window=torch.hann_window(window_length=512).to(device))
        fft_pred = torch.stft(signal_pred, 512, window=torch.hann_window(window_length=512).to(device))

        fft_target = fft_target * fft_target
        fft_pred = fft_pred * fft_pred
        a = fft_target.split(1, dim=2)[0]
        b = fft_target.split(1, dim=2)[1]
        fft_target = a + b
        a = fft_pred.split(1, dim=2)[0]
        b = fft_pred.split(1, dim=2)[1]
        fft_pred = a + b

        fft_target = torch.log(torch.sqrt(fft_target)) / -10
        fft_pred = torch.log(torch.sqrt(fft_pred)) / -10
        fft_target = torch.clamp(fft_target, min=1e-5, max=1.0 - 1e-5)
        fft_pred = torch.clamp(fft_pred, min=1e-5, max=1.0 - 1e-5)
        #from ipdb import set_trace
        #set_trace()
        loss = -(fft_target * torch.log(fft_pred) + (1.0 - fft_target) * torch.log(1.0 - fft_pred))
        loss = loss.sum() / (fft_target.shape[0] * fft_target.shape[1])
        return loss

    def learn(self, wave, mgc, batch_size):
        last_proc = 0
        total_loss = 0
        num_batches = 0
        # batch_size = batch_size * self.UPSAMPLE_COUNT
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
                mgc_list.append(mgc[mgc_index])
                for ii in range(self.UPSAMPLE_COUNT):
                    signal.append(wave[mgc_index * self.UPSAMPLE_COUNT + ii])

                if len(mgc_list) == batch_size:
                    self.trainer.zero_grad()
                    num_batches += 1
                    y_pred, mean, logvar, weight, y_aux = self.network(mgc_list, signal=signal)
                    # disc, cont = self.dio.ulaw_encode(signal[self.RECEPTIVE_SIZE:])
                    # from ipdb import set_trace
                    # set_trace()
                    y_target = torch.tensor(signal[self.RECEPTIVE_SIZE:], dtype=torch.float).to(device)

                    loss = self._compute_mixture_loss(y_target, mean, logvar,
                                                      weight)  # self.cross_loss(y_softmax, y_target)

                    loss += self._compute_aux_loss(y_target, y_aux)
                    total_loss += loss
                    loss.backward()
                    self.trainer.step()

                    mgc_list = []
                    signal = signal[-self.RECEPTIVE_SIZE:]

        total_loss = total_loss.item()
        # self.cnt += 1
        return total_loss / num_batches


class VocoderNetwork(nn.Module):
    def __init__(self, receptive_field=512, mgc_size=60, upsample_size=200, num_mixtures=10):
        super(VocoderNetwork, self).__init__()

        self.RECEPTIVE_FIELD = receptive_field
        self.NUM_NETWORKS = 1
        self.MGC_SIZE = mgc_size
        self.UPSAMPLE_SIZE = upsample_size
        self.NUM_MIXTURES = num_mixtures

        self.convolutions = FullNet(self.RECEPTIVE_FIELD, mgc_size, 256)

        self.conditioning = nn.Sequential(nn.ConvTranspose2d(1, 1, (5, 2), padding=(2, 0), stride=(1, 2)), nn.ELU(),
                                          nn.ConvTranspose2d(1, 1, (5, 5), padding=(2, 0), stride=(1, 5)), nn.ELU(),
                                          nn.ConvTranspose2d(1, 1, (5, 5), padding=(2, 0), stride=(1, 5)), nn.ELU(),
                                          nn.ConvTranspose2d(1, 1, (5, 4), padding=(2, 0), stride=(1, 4)), nn.ELU())

        # self.softmax_layer = nn.Linear(64, 256)
        self.pre_output = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, num_mixtures)
        self.stdev_layer = nn.Linear(256, num_mixtures)
        self.logit_layer = nn.Linear(256, num_mixtures)
        self.conditioning_out = nn.Linear(mgc_size, 1)

        self.act = nn.Softmax(dim=1)
        self.softsign = torch.nn.Softsign()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, mgc, signal=None, prev=None, training=False):
        # x = x.reshape(x.shape[0], 1, x.shape[1])

        if signal is not None:
            # prepare the input

            x_list = []
            for ii in range(len(signal) - self.RECEPTIVE_FIELD):
                x_list.append(signal[ii:ii + self.RECEPTIVE_FIELD])

            x = torch.Tensor(x_list).to(device)
            x = x.reshape(x.shape[0], 1, x.shape[1])
            pre_softmax = []

            # from ipdb import set_trace
            # set_trace()
            conditioning = self.conditioning(torch.Tensor(mgc).to(device).reshape(len(mgc), 1, 1, self.MGC_SIZE))
            conditioning = conditioning.reshape(len(mgc) * self.UPSAMPLE_SIZE, self.MGC_SIZE)

            pre = self.convolutions(x, conditioning)
            pre = pre.reshape(pre.shape[0], pre.shape[1])
            pre = torch.relu(self.pre_output(pre))

            # from ipdb import set_trace
            # set_trace()

            # softmax = self.softmax_layer(pre)  # self.act()
            mean = self.mean_layer(pre)
            stdev = self.stdev_layer(pre)
            logit = self.logit_layer(pre)
            conditioning_out = self.softsign(self.conditioning_out(conditioning))
            # from ipdb import set_trace
            # set_trace()
        else:
            signal = prev[-self.RECEPTIVE_FIELD:]
            for zz in range(len(mgc)):
                conditioning = self.conditioning(torch.Tensor(mgc[zz]).to(device).reshape(1, 1, 1, self.MGC_SIZE))
                conditioning = conditioning.reshape(self.UPSAMPLE_SIZE, self.MGC_SIZE)
                for ii in range(self.UPSAMPLE_SIZE):
                    x = torch.Tensor(signal[-self.RECEPTIVE_FIELD:]).to(device)
                    x = x.reshape(1, 1, x.shape[0])
                    # cond = self.conditioning(torch.Tensor(mgc[ii]).to(device).reshape(1, 60))
                    pre = self.convolutions(x, conditioning[ii].reshape(1, self.MGC_SIZE))

                    pre = pre.reshape(pre.shape[0], pre.shape[1])
                    pre = torch.relu(self.pre_output(pre))
                    # softmax = self.act(self.softmax_layer(pre))
                    # from ipdb import set_trace
                    # set_trace()
                    # sample = self._pick_sample(softmax.data.cpu().numpy().reshape(256), temperature=0.8)
                    mean = self.mean_layer(pre)
                    stdev = self.stdev_layer(pre)
                    logit = self.logit_layer(pre)
                    sample = self._pick_sample_from_logistics(mean, stdev, logit)
                    # f = float(sample) / 128 - 1.0
                    # sign = np.sign(f)
                    # decoded = sign * (1.0 / 255.0) * (pow(1.0 + 255, abs(f)) - 1.0)
                    signal.append(sample)
                conditioning_out = None

        return signal[self.RECEPTIVE_FIELD:], mean, stdev, logit, conditioning_out  # softmax

    def _pick_sample_from_logistics(self, mean, stdev, logit_probs):
        log_scale_min = -7.0
        temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = logit_probs.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)

        # (B, T) -> (B, T, nr_mix)
        one_hot = to_one_hot(argmax, self.NUM_MIXTURES)
        # select logistic parameters
        means = torch.sum(mean * one_hot, dim=-1)
        log_scales = torch.clamp(torch.sum(
            stdev * one_hot, dim=-1), min=log_scale_min)
        # sample from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

        x = torch.clamp(torch.clamp(x, min=-1.), max=1.)
        return x

    def _pick_sample(self, probs, temperature=1.0):
        probs = probs / np.sum(probs)
        scaled_prediction = np.log(probs) / temperature
        scaled_prediction = (scaled_prediction -
                             np.logaddexp.reduce(scaled_prediction))
        scaled_prediction = np.exp(scaled_prediction)
        # print np.sum(probs)
        # probs = probs / np.sum(probs)
        return np.random.choice(np.arange(256), p=scaled_prediction)


class CondConv(nn.Module):
    def __init__(self, input_size, output_size, cond_size, kernel_size, stride):
        super(CondConv, self).__init__()
        self.conv_input = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=0,
                                    bias=False)
        self.conv_gate = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=0,
                                   bias=False)
        self.conv_residual = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=0,
                                       bias=False)
        self.cond_input = nn.Linear(cond_size, output_size, bias=False)
        self.cond_gate = nn.Linear(cond_size, output_size, bias=False)

        torch.nn.init.xavier_uniform_(self.conv_input.weight)
        torch.nn.init.xavier_uniform_(self.conv_gate.weight)
        torch.nn.init.xavier_uniform_(self.conv_residual.weight)
        torch.nn.init.xavier_uniform_(self.cond_input.weight)
        torch.nn.init.xavier_uniform_(self.cond_gate.weight)

    def forward(self, conv, cond):
        input = self.conv_input(conv)
        gate = self.conv_gate(conv)
        residual = self.conv_residual(conv)

        # from ipdb import set_trace
        # set_trace()
        input_cond = self.cond_input(cond)
        gate_cond = self.cond_gate(cond)
        input_cond = input_cond.reshape(input_cond.shape[0], input_cond.shape[1], 1).expand(-1, -1, input.shape[2])
        gate_cond = gate_cond.reshape(input_cond.shape[0], input_cond.shape[1], 1).expand(-1, -1, input.shape[2])
        it = torch.tanh(input + input_cond)
        gt = torch.sigmoid(gate + gate_cond)
        output = it * gt + residual
        return output


class FullNet(nn.Module):
    def __init__(self, receptive_field, conditioning_size, filter_size):
        super(FullNet, self).__init__()
        self.RECEPTIVE_FIELD = receptive_field
        self.FILTER_SIZE = filter_size
        self.layers = torch.nn.ModuleList([CondConv(1, filter_size, conditioning_size, kernel_size=2, stride=2),
                                           CondConv(filter_size, filter_size, conditioning_size, kernel_size=2,
                                                    stride=2),
                                           CondConv(filter_size, filter_size, conditioning_size, kernel_size=2,
                                                    stride=2),
                                           CondConv(filter_size, filter_size, conditioning_size, kernel_size=2,
                                                    stride=2),
                                           CondConv(filter_size, filter_size, conditioning_size, kernel_size=2,
                                                    stride=2),
                                           CondConv(filter_size, filter_size, conditioning_size, kernel_size=2,
                                                    stride=2),
                                           CondConv(filter_size, filter_size, conditioning_size, kernel_size=2,
                                                    stride=2),
                                           CondConv(filter_size, filter_size, conditioning_size, kernel_size=2,
                                                    stride=2),
                                           CondConv(filter_size, filter_size, conditioning_size, kernel_size=2,
                                                    stride=2)])

    def forward(self, input, cond):
        layer_input = input
        for iLayer in range(9):
            layer_input = self.layers[iLayer](layer_input, cond)
        # from ipdb import set_trace
        # set_trace()
        return layer_input
