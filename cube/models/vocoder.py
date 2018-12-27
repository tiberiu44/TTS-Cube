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


class ParallelWavenetVocoder:
    def __init__(self, params, wavenet):
        self.params = params
        self.UPSAMPLE_COUNT = int(12.5 * params.target_sample_rate / 1000)
        self.RECEPTIVE_SIZE = 512  # this means 16ms
        self.MGC_SIZE = params.mgc_order
        self.dio = DatasetIO()
        self.vocoder = MelVocoder()
        self.wavenet = wavenet
        self.network = ParallelVocoderNetwork(receptive_field=self.RECEPTIVE_SIZE, filter_size=64).to(device)
        self.trainer = torch.optim.Adam(self.network.parameters(), lr=self.params.learning_rate)

    def synthesize(self, mgc, batch_size):
        last_proc = 0
        synth = []

        noise = np.random.normal(0, 1, (self.RECEPTIVE_SIZE + len(mgc) * self.UPSAMPLE_COUNT))
        noise = np.array(noise, dtype=np.float32)

        for mgc_index in range(len(mgc)):
            curr_proc = int((mgc_index + 1) * 100 / len(mgc))
            if curr_proc % 5 == 0 and curr_proc != last_proc:
                while last_proc < curr_proc:
                    last_proc += 5
                    sys.stdout.write(' ' + str(last_proc))
                    sys.stdout.flush()

            input = mgc[mgc_index]
            start = mgc_index * self.UPSAMPLE_COUNT
            stop = mgc_index * self.UPSAMPLE_COUNT + self.RECEPTIVE_SIZE + self.UPSAMPLE_COUNT
            [signal, means, logvars] = self.network([input], noise[start:stop])
            for zz in signal:
                synth.append(zz.item())

        synth = np.array(synth, dtype=np.float32)
        synth = np.clip(synth * 32768, -32767, 32767)
        synth = np.array(synth, dtype=np.int16)

        return synth

    def learn(self, wave, mgc, batch_size):
        last_proc = 0
        total_loss = 0
        total_loss_iaf = 0
        total_loss_power = 0
        num_batches = 0

        mgc_list = []
        signal = [0 for ii in range(self.RECEPTIVE_SIZE)]
        noise = np.random.normal(0, 1, (self.RECEPTIVE_SIZE + len(mgc) * self.UPSAMPLE_COUNT))
        noise = np.array(noise, dtype=np.float32)

        start = 0

        for mgc_index in range(len(mgc)):
            curr_proc = int((mgc_index + 1) * 100 / len(mgc))
            if curr_proc % 10 == 0 and curr_proc != last_proc:
                while last_proc < curr_proc:
                    last_proc += 10
                    sys.stdout.write(' ' + str(last_proc))
                    sys.stdout.flush()
            if mgc_index < len(mgc) - 1:
                mgc_list.append(mgc[mgc_index])

                if len(mgc_list) == batch_size:
                    self.trainer.zero_grad()
                    num_batches += 1

                    stop = start + len(mgc_list) * self.UPSAMPLE_COUNT + self.RECEPTIVE_SIZE
                    # from ipdb import set_trace
                    # set_trace()
                    y_pred, mean, logvar = self.network(mgc_list, noise[start:stop])
                    for zz in y_pred:
                        signal.append(zz.item())

                    t_mean, t_logvar, t_logits = self._compute_wavenet_target(signal[start:stop], mgc_list)
                    loss, loss_iaf, loss_power = self._compute_iaf_loss(y_pred, mean, logvar, t_mean, t_logvar,
                                                                        t_logits,
                                                                        torch.tensor(
                                                                            wave[start:stop - self.RECEPTIVE_SIZE]).to(
                                                                            device))
                    start = stop - self.RECEPTIVE_SIZE

                    total_loss += loss
                    total_loss_power += loss_power
                    total_loss_iaf += loss_iaf
                    loss.backward()
                    self.trainer.step()

                    mgc_list = []

        total_loss = total_loss.item()
        # self.cnt += 1
        total_loss_iaf = total_loss_iaf.item() / num_batches
        total_loss_power = total_loss_power.item() / num_batches

        sys.stdout.write(" iaf=" + str(total_loss_iaf) + ", power=" + str(total_loss_power) + " ")

        return total_loss / num_batches

    def _compute_wavenet_target(self, signal, mgc):
        with torch.no_grad():
            # prepare the input

            x_list = []
            for ii in range(len(signal) - self.RECEPTIVE_SIZE):
                x_list.append(signal[ii:ii + self.RECEPTIVE_SIZE])

            x = torch.Tensor(x_list).to(device)

            x = x.reshape(x.shape[0], 1, x.shape[1])

            conditioning = self.wavenet.network.conditioning(
                torch.Tensor(mgc).to(device).reshape(len(mgc), 1, 1, self.MGC_SIZE))
            conditioning = conditioning.reshape(len(mgc) * self.UPSAMPLE_COUNT, self.MGC_SIZE)

            pre = self.wavenet.network.convolutions(x, conditioning)
            pre = pre.reshape(pre.shape[0], pre.shape[1])
            pre = torch.relu(self.wavenet.network.pre_output(pre))

            mean = self.wavenet.network.mean_layer(pre)
            stdev = self.wavenet.network.stdev_layer(pre)
            logits = self.wavenet.network.logit_layer(pre)
            return mean, stdev, logits

    def _compute_iaf_loss(self, p_y, p_mean, p_logvar, t_mean, t_logvar, t_logits, t_y):

        log_scale_min = -7.0

        temp = t_logits.data.new(t_logits.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = t_logits.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)

        one_hot = to_one_hot(argmax, self.wavenet.network.NUM_MIXTURES)
        sel_mean = torch.sum(t_mean * one_hot, dim=-1)
        sel_logvar = torch.clamp(torch.sum(
            t_logvar * one_hot, dim=-1), min=log_scale_min)

        # from ipdb import set_trace
        # set_trace()

        p_mean = p_mean.reshape((p_mean.shape[0]))
        p_logvar = torch.clamp(p_logvar.reshape((p_logvar.shape[0])), min=log_scale_min)
        m0 = p_mean
        m1 = sel_mean.detach()
        logv0 = p_logvar
        logv1 = sel_logvar.detach()
        v0 = torch.exp(logv0)
        v1 = torch.exp(logv1)
        # from ipdb import set_trace
        # set_trace()
        #loss_iaf = torch.mean(
        #    logv1 - logv0 + (torch.pow(v0, 2) + torch.pow(m0 - m1, 2)) / (2.0 * torch.pow(v1, 2)) - 0.5)

        # loss_iaf = torch.sum(torch.pow(m1 - m0, 2) + torch.pow(logv1 - logv0, 2)) / p_y.shape[0]
        # loss_iaf1 = torch.mean(4 * torch.pow(logv1 - logv0, 2))
        # loss_iaf2 = torch.sum(torch.log(v1 / v0) \
        #                      + (torch.pow(v0, 2) - torch.pow(v1, 2)
        #                         + torch.pow((m0 - m1), 2)) / (2 * torch.pow(v1, 2))) / p_y.shape[0]
        # loss_iaf = loss_iaf1 + loss_iaf2
        # prob_mean=1.0-np.tanh(np.abs(x-m)/(2*s))

        prob_mean_m0 = torch.clamp(1.0 - torch.tanh(torch.abs(m1 - m0) / (2 * torch.exp(logv0))), 1e-8, 1.0 - 1e-8)
        prob_mean_m1 = torch.clamp(1.0 - torch.tanh(torch.abs(m0 - m1) / (2 * torch.exp(logv1))), 1e-8, 1.0 - 1e-8)
        loss_iaf = torch.mean(-torch.log(prob_mean_m0) - torch.log(prob_mean_m1) + (logv0 + 2.0))

        fft_orig = torch.stft(t_y.reshape(t_y.shape[0]), n_fft=512,
                              window=torch.hann_window(window_length=512).to(device))
        fft_pred = torch.stft(p_y.reshape(p_y.shape[0]), n_fft=512,
                              window=torch.hann_window(window_length=512).to(device))
        real_orig = fft_orig[:, :, 0]
        im_org = fft_orig[:, :, 1]
        power_orig = torch.sqrt(torch.pow(real_orig, 2) + torch.pow(im_org, 2))
        real_pred = fft_pred[:, :, 0]
        im_pred = fft_pred[:, :, 1]
        power_pred = torch.sqrt(torch.pow(real_pred, 2) + torch.pow(im_pred, 2))
        loss_power1 = torch.sum(torch.pow(torch.norm(torch.abs(power_pred) - torch.abs(power_orig), p=2, dim=1), 2)) / (
                power_pred.shape[0] * power_pred.shape[1])

        # freq1 = int(3000 / (self.params.target_sample_rate * 0.5) * 257)
        # fft_pred = torch.stft(p_y.reshape(p_y.shape[0]), n_fft=512,
        #                       window=torch.hann_window(window_length=512).to(device))[freq1:, :]
        # fft_orig = torch.stft(t_y.reshape(t_y.shape[0]), n_fft=512,
        #                       window=torch.hann_window(window_length=512).to(device))[freq1:, :]
        # real_orig = fft_orig[:, :, 0]
        # im_org = fft_orig[:, :, 1]
        # power_orig = torch.sqrt(torch.pow(real_orig, 2) + torch.pow(im_org, 2))
        # real_pred = fft_pred[:, :, 0]
        # im_pred = fft_pred[:, :, 1]
        # power_pred = torch.sqrt(torch.pow(real_pred, 2) + torch.pow(im_pred, 2))
        #
        # loss_power2 = torch.mean(torch.pow(torch.norm(torch.abs(power_pred) - torch.abs(power_orig), p=2, dim=1), 2))
        # from ipdb import set_trace
        # set_trace()

        return loss_iaf + loss_power1, loss_iaf, loss_power1

    def store(self, output_base):
        torch.save(self.network.state_dict(), output_base + ".network")

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


class WavenetVocoder:
    def __init__(self, params, model=None, runtime=False):
        self.params = params

        self.UPSAMPLE_COUNT = int(12.5 * params.target_sample_rate / 1000)
        self.RECEPTIVE_SIZE = 512  # this means 16ms

        self.dio = DatasetIO()
        self.vocoder = MelVocoder()

        self.network = VocoderNetwork(receptive_field=self.RECEPTIVE_SIZE, filter_size=256).to(device)
        self.trainer = torch.optim.Adam(self.network.parameters(), lr=self.params.learning_rate)

    def synthesize(self, mgc, batch_size, sample=True, temperature=1.0, path=None, return_residual=False):
        last_proc = 0
        synth = [0 for ii in range(self.RECEPTIVE_SIZE)]

        for mgc_index in range(len(mgc)):
            curr_proc = int((mgc_index + 1) * 100 / len(mgc))
            if curr_proc % 5 == 0 and curr_proc != last_proc:
                while last_proc < curr_proc:
                    last_proc += 5
                    sys.stdout.write(' ' + str(last_proc))
                    sys.stdout.flush()

            input = mgc[mgc_index]

            [signal, means, logvars, logits] = self.network([input], prev=synth)

            #
            for zz in signal:
                synth.append(zz.item())

        synth = np.array(synth[self.RECEPTIVE_SIZE:], dtype=np.float32)
        synth = np.clip(synth * 32768, -32767, 32767)
        synth = np.array(synth, dtype=np.int16)

        return synth

    def store(self, output_base):
        torch.save(self.network.state_dict(), output_base + ".network")

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

    def _compute_mixture_loss(self, y_target, means, logvars, logit_probs):
        num_classes = 65536

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
                    y_pred, mean, logvar, weight = self.network(mgc_list, signal=signal)

                    y_target = torch.tensor(signal[self.RECEPTIVE_SIZE:], dtype=torch.float).to(device)

                    loss = self._compute_mixture_loss(y_target, mean, logvar,
                                                      weight)

                    total_loss += loss
                    loss.backward()
                    self.trainer.step()

                    mgc_list = []
                    signal = signal[-self.RECEPTIVE_SIZE:]

        total_loss = total_loss.item()
        # self.cnt += 1
        return total_loss / num_batches


class ParallelVocoderNetwork(nn.Module):
    def __init__(self, receptive_field=1024, mgc_size=60, upsample_size=200, filter_size=64):
        super(ParallelVocoderNetwork, self).__init__()

        self.RECEPTIVE_FIELD = receptive_field
        self.NUM_NETWORKS = 1
        self.MGC_SIZE = mgc_size
        self.UPSAMPLE_SIZE = upsample_size

        self.convolutions = WaveNet(self.RECEPTIVE_FIELD, mgc_size, filter_size)

        self.conditioning = nn.Sequential(nn.Linear(mgc_size, mgc_size * upsample_size), nn.Tanh())

        self.pre_output = nn.Linear(filter_size, 256)
        self.mean_layer = nn.Linear(256, 1)
        self.stdev_layer = nn.Linear(256, 1)
        self.conditioning_out = nn.Linear(mgc_size, 1)

        self.act = nn.Softmax(dim=1)
        self.softsign = torch.nn.Softsign()

    def reparameterize(self, mu, logvar, rand):
        std = torch.exp(0.5 * logvar)
        eps = rand
        return eps.mul(std).add_(mu)

    def forward(self, mgc, noise):
        # prepare the input
        x_list = []
        for ii in range(len(noise) - self.RECEPTIVE_FIELD):
            x_list.append(noise[ii:ii + self.RECEPTIVE_FIELD])

        x = torch.Tensor(x_list).to(device)
        x = x.reshape(x.shape[0], 1, x.shape[1])

        conditioning = self.conditioning(torch.Tensor(mgc).to(device).reshape(len(mgc), 1, 1, self.MGC_SIZE))
        conditioning = conditioning.reshape(len(mgc) * self.UPSAMPLE_SIZE, self.MGC_SIZE)

        pre = self.convolutions(x, conditioning)
        pre = pre.reshape(pre.shape[0], pre.shape[1])
        pre = torch.relu(self.pre_output(pre))

        mean = self.mean_layer(pre)
        logvar = self.stdev_layer(pre)

        return self.reparameterize(mean, logvar,
                                   torch.tensor(noise[self.RECEPTIVE_FIELD:]).to(device).reshape(mean.shape[0],
                                                                                                 mean.shape[1])), \
               mean, logvar


class VocoderNetwork(nn.Module):
    def __init__(self, receptive_field=1024, mgc_size=60, upsample_size=200, num_mixtures=10, filter_size=256):
        super(VocoderNetwork, self).__init__()

        self.RECEPTIVE_FIELD = receptive_field
        self.NUM_NETWORKS = 1
        self.MGC_SIZE = mgc_size
        self.UPSAMPLE_SIZE = upsample_size
        self.NUM_MIXTURES = num_mixtures

        self.convolutions = WaveNet(self.RECEPTIVE_FIELD, mgc_size, filter_size)

        self.conditioning = nn.Sequential(nn.Linear(mgc_size, mgc_size * upsample_size), nn.Tanh())

        # self.softmax_layer = nn.Linear(64, 256)
        self.pre_output = nn.Linear(filter_size, 256)
        self.mean_layer = nn.Linear(256, num_mixtures)
        self.stdev_layer = nn.Linear(256, num_mixtures)
        self.logit_layer = nn.Linear(256, num_mixtures)
        self.conditioning_out = nn.Linear(mgc_size, 1)

        self.act = nn.Softmax(dim=1)
        self.softsign = torch.nn.Softsign()

    def forward(self, mgc, signal=None, prev=None, training=False):
        if signal is not None:
            # prepare the input

            x_list = []
            for ii in range(len(signal) - self.RECEPTIVE_FIELD):
                x_list.append(signal[ii:ii + self.RECEPTIVE_FIELD])

            x = torch.Tensor(x_list).to(device)
            x = x.reshape(x.shape[0], 1, x.shape[1])

            conditioning = self.conditioning(torch.Tensor(mgc).to(device).reshape(len(mgc), 1, 1, self.MGC_SIZE))
            conditioning = conditioning.reshape(len(mgc) * self.UPSAMPLE_SIZE, self.MGC_SIZE)

            pre = self.convolutions(x, conditioning)
            pre = pre.reshape(pre.shape[0], pre.shape[1])
            pre = torch.relu(self.pre_output(pre))

            mean = self.mean_layer(pre)
            stdev = self.stdev_layer(pre)
            logits = self.logit_layer(pre)
        else:
            signal = prev[-self.RECEPTIVE_FIELD:]
            for zz in range(len(mgc)):
                conditioning = self.conditioning(torch.Tensor(mgc[zz]).to(device).reshape(1, 1, 1, self.MGC_SIZE))
                conditioning = conditioning.reshape(self.UPSAMPLE_SIZE, self.MGC_SIZE)
                for ii in range(self.UPSAMPLE_SIZE):
                    x = torch.Tensor(signal[-self.RECEPTIVE_FIELD:]).to(device)
                    x = x.reshape(1, 1, x.shape[0])

                    pre = self.convolutions(x, conditioning[ii].reshape(1, self.MGC_SIZE))
                    pre = pre.reshape(pre.shape[0], pre.shape[1])
                    pre = torch.relu(self.pre_output(pre))

                    mean = self.mean_layer(pre)
                    stdev = self.stdev_layer(pre)
                    logits = self.logit_layer(pre)
                    sample = self._pick_sample_from_logistics(mean, stdev, logits)
                    signal.append(sample)

        return signal[self.RECEPTIVE_FIELD:], mean, stdev, logits

    def _pick_sample_from_logistics(self, mean, stdev, logit_probs):
        log_scale_min = -7.0
        temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = logit_probs.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)

        one_hot = to_one_hot(argmax, self.NUM_MIXTURES)
        means = torch.sum(mean * one_hot, dim=-1)
        log_scales = torch.clamp(torch.sum(
            stdev * one_hot, dim=-1), min=log_scale_min)
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
        return np.random.choice(np.arange(256), p=scaled_prediction)


class CondConv(nn.Module):
    def __init__(self, input_size, output_size, cond_size, kernel_size, stride):
        super(CondConv, self).__init__()
        self.conv_input = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=0,
                                    bias=True)
        self.conv_gate = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=0,
                                   bias=True)
        self.conv_residual = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=0,
                                       bias=True)
        self.cond_input = nn.Linear(cond_size, output_size, bias=True)
        self.cond_gate = nn.Linear(cond_size, output_size, bias=True)

        torch.nn.init.xavier_normal_(self.conv_input.weight)
        torch.nn.init.xavier_normal_(self.conv_gate.weight)
        torch.nn.init.xavier_normal_(self.conv_residual.weight)
        torch.nn.init.xavier_normal_(self.cond_input.weight)
        torch.nn.init.xavier_normal_(self.cond_gate.weight)

    def forward(self, conv, cond):
        input = self.conv_input(conv)
        gate = self.conv_gate(conv)
        residual = self.conv_residual(conv)

        input_cond = self.cond_input(cond)
        gate_cond = self.cond_gate(cond)
        input_cond = input_cond.reshape(input_cond.shape[0], input_cond.shape[1], 1).expand(-1, -1, input.shape[2])
        gate_cond = gate_cond.reshape(input_cond.shape[0], input_cond.shape[1], 1).expand(-1, -1, input.shape[2])
        it = torch.tanh(input + input_cond)
        gt = torch.sigmoid(gate + gate_cond)
        output = (it * gt + residual) * 0.70710678118  # don't know why :)
        return output


class WaveNet(nn.Module):
    def __init__(self, receptive_field, conditioning_size, filter_size):
        super(WaveNet, self).__init__()
        self.RECEPTIVE_FIELD = receptive_field
        self.FILTER_SIZE = filter_size
        ml = [CondConv(1, filter_size, conditioning_size, kernel_size=2, stride=2)]
        rc = receptive_field // 2
        while rc != 1:
            ml.append(CondConv(filter_size, filter_size, conditioning_size, kernel_size=2,
                               stride=2))
            rc = rc // 2
        self.num_layers = len(ml)
        self.layers = torch.nn.ModuleList(ml)

    def forward(self, input, cond):
        layer_input = input
        for iLayer in range(self.num_layers):
            layer_input = self.layers[iLayer](layer_input, cond)

        return layer_input
