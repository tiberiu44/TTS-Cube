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
        mgc_list = []
        tail = None
        for mgc_index in range(len(mgc)):
            mgc_list.append(mgc[mgc_index])
            curr_proc = int((mgc_index + 1) * 100 / len(mgc))
            if curr_proc % 5 == 0 and curr_proc != last_proc:
                while last_proc < curr_proc:
                    last_proc += 5
                    sys.stdout.write(' ' + str(last_proc))
                    sys.stdout.flush()
            if len(mgc_list) == batch_size:
                start = (mgc_index - batch_size + 1) * self.UPSAMPLE_COUNT
                stop = start + len(mgc_list) * self.UPSAMPLE_COUNT + self.RECEPTIVE_SIZE
                with torch.no_grad():
                    [signal, means, logvars, tail] = self.network(mgc_list, noise[start:stop], tail=tail)
                    for zz in signal:
                        synth.append(zz.item())

                mgc_list = []
        if len(mgc_list) != 0:
            start = (mgc_index - batch_size + 1) * self.UPSAMPLE_COUNT
            stop = start + len(mgc_list) * self.UPSAMPLE_COUNT + self.RECEPTIVE_SIZE
            with torch.no_grad():
                [signal, means, logvars, tail] = self.network(mgc_list, noise[start:stop], tail=tail)
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
        tail = None
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
                    y_pred, mean, logvar, tail = self.network(mgc_list, noise[start:stop], tail=tail)

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

    def _compute_loss_by_sampling(self, p_mean, p_logvar, t_mean, t_logvar):
        # generate random samples and estimate their probability using student and teacher
        loss = 0
        NUM_SAMPLES = 20
        std = torch.exp(0.5 * t_logvar)
        for ii in range(NUM_SAMPLES):
            eps = torch.randn_like(t_mean)
            samples = eps.mul(std).add_(t_mean)
            prob_t = torch.clamp(1.0 - torch.tanh(torch.abs(samples - t_mean) / (2 * torch.exp(t_logvar))), 1e-5,
                                 1.0 - 1e-5)
            prob_p = torch.clamp(1.0 - torch.tanh(torch.abs(samples - p_mean) / (2 * torch.exp(p_logvar))), 1e-5,
                                 1.0 - 1e-5)
            loss += torch.mean(-prob_t * torch.log(prob_p) - (1.0 - prob_t) * torch.log(1.0 - prob_p))

        return loss / NUM_SAMPLES

    def _kl_loss(self, mu_q, logs_q, mu_p, logs_p, log_std_min=-7.0):
        # KL (q || p)
        # q ~ N(mu_q, logs_q.exp_()), p ~ N(mu_p, logs_p.exp_())
        logs_q = torch.clamp(logs_q, min=log_std_min)
        logs_p = torch.clamp(logs_p, min=log_std_min)
        kl_loss = (logs_p - logs_q) + 0.5 * (
                (torch.exp(2. * logs_q) + torch.pow(mu_p - mu_q, 2)) * torch.exp(-2. * logs_p) - 1.)
        reg_loss = torch.pow(logs_q - logs_p, 2)
        return kl_loss + 4.0 * reg_loss

    def _power_loss(self, p_y, t_y):
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
        return torch.sum(torch.pow(torch.norm(torch.abs(power_pred) - torch.abs(power_orig), p=2, dim=1), 2)) / (
                power_pred.shape[0] * power_pred.shape[1])

    def _compute_iaf_loss(self, p_y, p_mean, p_logvar, t_mean, t_logvar, t_logits, t_y):

        log_scale_min = -7.0

        temp = t_logits.data.new(t_logits.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = t_logits.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)

        one_hot = to_one_hot(argmax, self.wavenet.network.NUM_MIXTURES)
        sel_mean = torch.sum(t_mean * one_hot, dim=-1).detach()
        sel_logvar = torch.clamp(torch.sum(
            t_logvar * one_hot, dim=-1), min=log_scale_min).detach()

        p_mean = p_mean.reshape((p_mean.shape[0]))
        p_logvar = p_logvar.reshape((p_logvar.shape[0]))

        # loss V1
        m0 = p_mean
        m1 = sel_mean
        logv0 = p_logvar
        logv1 = sel_logvar
        loss_teacher1 = torch.sum(torch.pow(m1 - m0, 2) + torch.pow(logv1 - logv0, 2)) / p_y.shape[0]
        loss_teacher2 = self._compute_loss_by_sampling(p_mean, p_logvar, m1, logv1)
        loss_teacher = loss_teacher1 + loss_teacher2

        # loss_V2
        # loss_teacher = torch.mean(self._kl_loss(sel_mean, sel_logvar, p_mean, p_logvar))

        loss_power = self._power_loss(p_y, t_y)

        return loss_teacher + loss_power, loss_teacher, loss_power

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

    def _compute_gaussian_loss(self, target, mean, logvar):
        log_std_min = -7
        logvar = torch.clamp(logvar, min=log_std_min)
        import math
        return torch.mean(
            -0.5 * (- math.log(2.0 * math.pi) - 2. * logvar - torch.pow(target - mean, 2) * torch.exp((-2.0 * logvar))))

    def learn(self, wave, mgc, batch_size):
        last_proc = 0
        total_loss = 0
        num_batches = 0
        # batch_size = batch_size * self.UPSAMPLE_COUNT
        tail = None
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
                    y_pred, mean, logvar, tail = self.network(mgc_list, signal=signal, tail=tail)

                    y_target = torch.tensor(signal[self.RECEPTIVE_SIZE:], dtype=torch.float).to(device)

                    loss = self._compute_gaussian_loss(y_target, mean, logvar)
                    # loss = self._compute_mixture_loss(y_target, mean, logvar,
                    #                                  weight)

                    total_loss += loss
                    loss.backward()
                    self.trainer.step()

                    mgc_list = []
                    signal = signal[-self.RECEPTIVE_SIZE:]

        total_loss = total_loss.item()
        # self.cnt += 1
        return total_loss / num_batches


class ParallelVocoderNetwork(nn.Module):
    def __init__(self, receptive_field=1024, mgc_size=60, upsample_size=200, filter_size=64, num_stacks=4):
        super(ParallelVocoderNetwork, self).__init__()

        self.RECEPTIVE_FIELD = receptive_field
        self.NUM_NETWORKS = 1
        self.MGC_SIZE = mgc_size
        self.UPSAMPLE_SIZE = upsample_size
        self.NUM_STACKS = num_stacks

        self.convolutions = torch.nn.ModuleList(
            [WaveNet(self.RECEPTIVE_FIELD, mgc_size, filter_size) for ii in range(num_stacks)])

        self.conditioning = nn.Sequential(nn.Linear(mgc_size, mgc_size * upsample_size), nn.Tanh())

        self.pre_output = torch.nn.ModuleList([nn.Linear(filter_size, 256) for ii in range(num_stacks)])
        self.mean_layer = torch.nn.ModuleList([nn.Linear(256, 1) for ii in range(num_stacks)])
        self.stdev_layer = torch.nn.ModuleList([nn.Linear(256, 1) for ii in range(num_stacks)])

    def reparameterize(self, mu, logvar, rand):
        std = torch.exp(0.5 * logvar)
        eps = rand
        return eps.mul(std).add_(mu)

    def forward(self, mgc, noise, tail=None):
        conditioning = self.conditioning(torch.Tensor(mgc).to(device).reshape(len(mgc), 1, 1, self.MGC_SIZE))
        conditioning = conditioning.reshape(len(mgc) * self.UPSAMPLE_SIZE, self.MGC_SIZE)

        prepend = torch.tensor(noise[:self.RECEPTIVE_FIELD]).to(device)
        prev_x = torch.tensor(noise).to(device)
        new_tail = []
        for iStack in range(self.NUM_STACKS):

            # prepare the input
            x_list = []
            for ii in range(len(prev_x) - self.RECEPTIVE_FIELD):
                x_list.append(prev_x[ii:ii + self.RECEPTIVE_FIELD])
            # from ipdb import set_trace
            # set_trace()
            x = torch.stack(x_list)
            x = x.reshape(len(mgc) * self.UPSAMPLE_SIZE, 1, x_list[0].shape[0])

            pre = self.convolutions[iStack](x, conditioning)
            pre = pre.reshape(pre.shape[0], pre.shape[1])
            pre = torch.relu(self.pre_output[iStack](pre))

            mean = self.mean_layer[iStack](pre)
            logvar = self.stdev_layer[iStack](pre)

            prev_x = self.reparameterize(mean, logvar,
                                         prev_x[self.RECEPTIVE_FIELD:].reshape(mean.shape[0], mean.shape[1]))
            new_tail.append(prev_x[-self.RECEPTIVE_FIELD:].detach())

            if iStack != self.NUM_STACKS - 1:
                if tail is None:
                    prev_x = torch.cat((prepend, prev_x.reshape((prev_x.shape[0]))))
                else:
                    prev_x = torch.cat((tail[iStack].reshape(tail[iStack].shape[0]), prev_x.reshape((prev_x.shape[0]))))

        return prev_x, mean, logvar, new_tail


class VocoderNetwork(nn.Module):
    def __init__(self, receptive_field=1024, mgc_size=60, upsample_size=200, filter_size=256,
                 num_blocks=4):
        super(VocoderNetwork, self).__init__()

        self.RECEPTIVE_FIELD = receptive_field
        self.NUM_NETWORKS = 1
        self.MGC_SIZE = mgc_size
        self.UPSAMPLE_SIZE = upsample_size
        self.NUM_BLOCKS = num_blocks

        self.convolutions = torch.nn.ModuleList(
            [WaveNet(self.RECEPTIVE_FIELD, mgc_size, filter_size) for ii in range(num_blocks)])

        self.conditioning = nn.Sequential(nn.Linear(mgc_size, mgc_size * upsample_size), nn.Tanh())

        self.pre_output = torch.nn.ModuleList([nn.Linear(filter_size, 256) for ii in range(num_blocks)])
        self.mean_layer = torch.nn.ModuleList([nn.Linear(256, 1) for ii in range(num_blocks)])
        self.stdev_layer = torch.nn.ModuleList([nn.Linear(256, 1) for ii in range(num_blocks)])

    def _reparameterize(self, mu, logvar, rand):
        std = torch.exp(0.5 * logvar)
        if rand is not None:
            eps = rand
        else:
            eps = torch.randn_like(mu).to(device)

        return eps.mul(std).add_(mu)

    def forward(self, mgc, signal=None, prev=None, training=False, tail=None):
        new_tail = []
        if signal is not None:
            # prepare the input

            conditioning = self.conditioning(torch.Tensor(mgc).to(device).reshape(len(mgc), 1, 1, self.MGC_SIZE))
            conditioning = conditioning.reshape(len(mgc) * self.UPSAMPLE_SIZE, self.MGC_SIZE)
            prev_x = torch.tensor(signal, dtype=torch.float32).to(device)
            for iStack in range(self.NUM_BLOCKS):
                x_list = []
                for ii in range(len(signal) - self.RECEPTIVE_FIELD):
                    x_list.append(prev_x[ii:ii + self.RECEPTIVE_FIELD])
                x = torch.cat(x_list).to(device)
                x = x.reshape(len(x_list), 1, self.RECEPTIVE_FIELD)

                pre = self.convolutions[iStack](x, conditioning)
                pre = pre.reshape(pre.shape[0], pre.shape[1])
                pre = torch.relu(self.pre_output[iStack](pre))

                mean = self.mean_layer[iStack](pre)
                stdev = self.stdev_layer[iStack](pre)
                new_x = self._reparameterize(mean, stdev, None)
                new_tail.append(new_x[-self.RECEPTIVE_FIELD:].detach())
                if tail is None:
                    prev_x = torch.cat((torch.tensor(signal[:self.RECEPTIVE_FIELD], dtype=torch.float32).to(device),
                                        torch.squeeze(new_x)))
                else:
                    # from ipdb import set_trace
                    # set_trace()
                    prev_x = torch.cat((torch.squeeze(tail[iStack]), torch.squeeze(new_x)))

        else:
            cnt = 0
            signal = torch.tensor(prev, dtype=torch.float32).to(device)
            for zz in range(len(mgc)):
                conditioning = self.conditioning(torch.Tensor(mgc[zz]).to(device).reshape(1, 1, 1, self.MGC_SIZE))
                conditioning = conditioning.reshape(self.UPSAMPLE_SIZE, self.MGC_SIZE)
                for ii in range(self.UPSAMPLE_SIZE):
                    for iStack in range(self.NUM_BLOCKS):
                        if iStack == 0:
                            prev_x = signal[-self.RECEPTIVE_FIELD:]
                        else:
                            if cnt >= self.RECEPTIVE_FIELD:
                                prev_x = new_tail[iStack][-self.RECEPTIVE_FIELD:]
                            else:
                                if tail is None:
                                    prev_x = torch.cat((signal[-self.RECEPTIVE_FIELD:][cnt + 1:], new_tail[iStack - 1][-cnt - 1:]))
                                else:
                                    prev_x = torch.cat((tail[iStack - 1][cnt + 1:], new_tail[iStack - 1][-cnt - 1:]))

                        if prev_x.shape[0] == 513:
                            from ipdb import set_trace
                            set_trace()
                        prev_x = prev_x.reshape(1, 1, self.RECEPTIVE_FIELD)
                        pre = self.convolutions[iStack](prev_x, conditioning[ii].reshape(1, self.MGC_SIZE))
                        pre = pre.reshape(pre.shape[0], pre.shape[1])
                        pre = torch.relu(self.pre_output[iStack](pre))

                        mean = self.mean_layer[iStack](pre)
                        stdev = self.stdev_layer[iStack](pre)
                        new_sample = self._reparameterize(mean, stdev, None)

                        if iStack == self.NUM_BLOCKS - 1:
                            signal = torch.cat((signal, new_sample.reshape((1))))
                        else:
                            if cnt == 0:
                                new_tail.append(new_sample.reshape((1)))
                            else:
                                new_tail[iStack] = torch.cat((new_tail[iStack], new_sample.reshape((1))))
                    cnt += 1
            for iStack in range(self.NUM_BLOCKS-1):
                new_tail[iStack] = new_tail[iStack][-self.RECEPTIVE_FIELD:].detach()

        return signal[self.RECEPTIVE_FIELD:], mean, stdev, new_tail

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
