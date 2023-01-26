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
import math
import numpy as np
import torch.nn.functional as F

from torch.distributions import Beta, Categorical
from torch.nn import CrossEntropyLoss


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


class GaussianOutput:
    def loss(self, y_hat, y, log_std_min=-14.0):
        assert y_hat.dim() == 3
        # assert y_hat.size(1) == 2

        ## (B x T x C)
        # y_hat = y_hat.transpose(1, 2)
        y = y.unsqueeze(2)
        mean = y_hat[:, :, :1]
        log_std = torch.clamp(y_hat[:, :, 1:], min=log_std_min)

        log_probs = -0.5 * (
                - math.log(2.0 * torch.pi) - 2. * log_std - torch.pow(y - mean, 2) * torch.exp((-2.0 * log_std)))
        return log_probs.squeeze().mean()

    def sample(self, y_hat, temperature=1.0):
        z = torch.randn((y_hat.shape[0], y_hat.shape[1], 1)) * 0.8
        return (y_hat[:, :, 0].unsqueeze(2) + z * torch.exp(y_hat[:, :, 1].unsqueeze(2))).squeeze(1)

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    @property
    def sample_size(self):
        return 2

    @property
    def stats(self):
        return 6e-6, 0.15


class BetaOutput:
    def loss(self, y_hat, y):
        loc_y = y_hat.exp()
        alpha = loc_y[:, :, 0].unsqueeze(-1)
        beta = loc_y[:, :, 1].unsqueeze(-1)
        dist = Beta(alpha, beta)
        # rescale y to be between
        y = (y + 1.0) / 2.0
        # note that we will get inf loss if y == 0 or 1.0 exactly, so we will clip it slightly just in case
        y = torch.clamp(y, 1e-5, 0.99999).unsqueeze(-1)
        # compute logprob
        loss = -dist.log_prob(y).squeeze(-1)
        return loss.mean()

    def sample(self, y_hat):
        output = torch.exp(y_hat)
        alfas = output[:, :, 0]
        betas = output[:, :, 1]

        # z = torch.randn((output.shape[0], output.shape[1]), device=self._get_device()) * 0.8
        # samples = means + z * torch.exp(logvars)
        distrib = Beta(alfas, betas)
        samples = (distrib.sample() - 0.5) * 2
        return samples

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    @property
    def sample_size(self):
        return 2

    @property
    def stats(self):
        return 6e-6, 0.15


class MOLOutput:
    def loss(self, y_hat, y, num_classes=65536, log_scale_min=None):
        if log_scale_min is None:
            log_scale_min = float(np.log(1e-14))
        assert y_hat.dim() == 3
        assert y_hat.shape[2] % 3 == 0
        nr_mix = y_hat.shape[2] // 3
        y = y.unsqueeze(2)

        # unpack parameters. (B, T, num_mixtures) x 3
        logit_probs = y_hat[:, :, :nr_mix]
        means = y_hat[:, :, nr_mix:2 * nr_mix]
        log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

        # B x T x 1 -> B x T x num_mixtures
        y = y.expand_as(means)

        centered_y = y - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
        cdf_min = torch.sigmoid(min_in)

        # log probability for edge case of 0 (before scaling)
        # equivalent: torch.log(F.sigmoid(plus_in))
        log_cdf_plus = plus_in - F.softplus(plus_in)

        # log probability for edge case of 255 (before scaling)
        # equivalent: (1 - F.sigmoid(min_in)).log()
        log_one_minus_cdf_min = -F.softplus(min_in)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        mid_in = inv_stdv * centered_y
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in our code)
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

        inner_inner_cond = (cdf_delta > 1e-5).float()

        inner_inner_out = inner_inner_cond * \
                          torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
                          (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
        inner_cond = (y > 0.999).float()
        inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
        cond = (y < -0.999).float()
        log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

        log_probs = log_probs + F.log_softmax(logit_probs, -1)

        return -torch.mean(log_sum_exp(log_probs))

    def sample(self, y, log_scale_min=None, temperature=1.0):
        """
        Sample from discretized mixture of logistic distributions
        Args:
            y (Tensor): B x T x C
            log_scale_min (float): Log scale minimum value
        Returns:
            Tensor: sample in range of [-1, 1].
        """
        if log_scale_min is None:
            log_scale_min = float(np.log(1e-14))
        assert y.shape[2] % 3 == 0
        nr_mix = y.shape[2] // 3

        # B x T x C
        # y = y.transpose(1, 2)
        logit_probs = y[:, :, :nr_mix]

        # sample mixture indicator from softmax
        temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1 - 1e-5) * temperature
        temp = logit_probs.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)

        # (B, T) -> (B, T, nr_mix)
        one_hot = F.one_hot(argmax, nr_mix).float()
        # select logistic parameters
        means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
        log_scales = torch.clamp(torch.sum(
            y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1), min=log_scale_min)
        # sample from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
        # u = means.data.new(means.size()).normal_(0, 1) * temperature
        # x = means + torch.exp(log_scales) * u

        x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

        return x

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    @property
    def sample_size(self):
        return 30

    @property
    def stats(self):
        return 6e-6, 0.15


class MULAWOutput:
    def __init__(self):
        self._loss = CrossEntropyLoss()

    def loss(self, y_hat, y):
        # y = (((y + 1.0) / 2) * 255).long()
        y = self.encode(y)
        return self._loss(y_hat.reshape(y_hat.shape[0] * y_hat.shape[1], -1), y.reshape(y.shape[0] * y.shape[1]))

    def sample(self, y):
        distrib = Categorical(logits=y)
        sample = distrib.sample()
        return self.decode(sample)
        # probs = torch.softmax(y, dim=-1)
        # from ipdb import set_trace
        # set_trace()
        # return self.decode(torch.argmax(y, dim=-1))

    def encode(self, x):
        quantization_channels = 256
        mu = quantization_channels - 1
        if isinstance(x, np.ndarray):
            x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).astype(int)
            x_mu = np.clip(x_mu, 0, quantization_channels - 1)
        elif isinstance(x, (torch.Tensor, torch.LongTensor)):

            if isinstance(x, torch.LongTensor):
                x = x.float()
            if x.get_device() != -1:
                mu = torch.FloatTensor([mu]).to(x.get_device())
            else:
                mu = torch.FloatTensor([mu])
            x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)

            x_mu = ((x_mu + 1) / 2 * mu + 0.5).long()
            x_mu = torch.clip(x_mu, 0, quantization_channels - 1)
        return x_mu

    def decode(self, x_mu):
        quantization_channels = 256
        mu = quantization_channels - 1.
        if isinstance(x_mu, np.ndarray):
            x = ((x_mu) / mu) * 2 - 1.
            x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
        elif isinstance(x_mu, (torch.Tensor, torch.LongTensor)):
            if isinstance(x_mu, (torch.LongTensor, torch.cuda.LongTensor)):
                x_mu = x_mu.float()
            mu = (torch.FloatTensor([mu]))
            x = ((x_mu) / mu) * 2 - 1.
            x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
        return x

    @property
    def sample_size(self):
        return 256

    @property
    def stats(self):
        return -0.019, 0.51


class RAWOutput:
    def __init__(self):
        self._loss = CrossEntropyLoss()

    def loss(self, y_hat, y):
        y = self.encode(y)
        return self._loss(y_hat.reshape(y_hat.shape[0] * y_hat.shape[1], -1), y.reshape(y.shape[0] * y.shape[1]))

    def sample(self, y):
        distrib = Categorical(logits=y)
        sample = distrib.sample()
        return self.decode(sample)

    def encode(self, x):
        y = torch.clip(((x + 1.0) / 2) * 255, 0, 255).long()
        return y

    def decode(self, x):
        x = ((x / 255) - 0.5) * 2
        return x

    @property
    def sample_size(self):
        return 256

    @property
    def stats(self):
        return -0.019, 0.15


if __name__ == '__main__':
    m = MULAWOutput()
    x_orig = np.array([1, 0.9, 0, -0.9, -1])
    print(x_orig)
    x_enc = m.encode(x_orig)
    print(x_enc)
    x_dec = m.decode(x_enc)
    print(x_dec)
