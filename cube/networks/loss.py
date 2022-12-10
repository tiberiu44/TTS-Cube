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

from torch.distributions import Beta


def gaussian_loss(y_hat, y, log_std_min=-7.0):
    assert y_hat.dim() == 3
    # assert y_hat.size(1) == 2

    ## (B x T x C)
    # y_hat = y_hat.transpose(1, 2)
    y = y.unsqueeze(2)
    mean = y_hat[:, :, :1]
    log_std = torch.clamp(y_hat[:, :, 1:], min=log_std_min)

    log_probs = -0.5 * (
            - math.log(2.0 * torch.pi) - 2. * log_std - torch.pow(y - mean, 2) * torch.exp((-2.0 * log_std)))
    return log_probs.squeeze()


def beta_loss(y_hat, y):
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
