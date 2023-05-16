import random

import torch
import torch.nn as nn

import torch
import json
import torch.nn as nn
import pytorch_lightning as pl
import sys
import itertools

from vector_quantize_pytorch import ResidualVQ

sys.path.append('')
sys.path.append('hifigan')
from cube.networks.modules import CubedallEncoder
from hifigan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from hifigan.env import AttrDict
from hifigan.meldataset import mel_spectrogram


class Cubedall(pl.LightningModule):
    def __init__(self, lr: float = 2e-4, train=True):
        super(Cubedall, self).__init__()
        self._current_lr = lr
        self._learning_rate = lr
        self._global_step = 0
        self._val_loss = 9999
        self._loaded_optimizer_states = None
        self._encoder = CubedallEncoder(32, 256, [4, 4, 5, 6])
        self._w = nn.Linear(1, 1)
        self._rvq = ResidualVQ(
            dim=256,
            codebook_size=1024,
            num_quantizers=6,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=2,
        )

        json_config = json.load(open('hifigan/config_v1-48khz.json'))
        h = AttrDict(json_config)
        self._generator = Generator(h)
        if train:
            self._mpd = MultiPeriodDiscriminator()
            self._msd = MultiScaleDiscriminator()

        self._loss_l1 = nn.L1Loss()
        self.automatic_optimization = False

    def forward(self, X):
        x = X['x']
        denoise = X['denoise']
        denoise = denoise.unsqueeze(1).repeat(1, 1, x.shape[1])
        x = x.unsqueeze(1)
        input = torch.cat([x, denoise], dim=1)
        # # compute conditioning using encoder and VQ
        # hidden = self._encoder(input).permute(0, 2, 1)
        # quantized, indices, loss_vq = self._rvq(hidden)
        # # generate using conditioning
        # y_g_hat = self._generator(quantized.permute(0, 2, 1))
        hidden = self._encoder(input).permute(0, 2, 1)
        # generate using conditioning
        y_g_hat = self._generator(hidden.permute(0, 2, 1))
        return y_g_hat

    def inference(self, X):
        x = X['x']
        denoise = X['denoise']
        denoise = denoise.unsqueeze(1).repeat(1, 1, x.shape[1])
        x = x.unsqueeze(1)
        input = torch.cat([x, denoise], dim=1)
        # # compute conditioning using encoder and VQ
        # hidden = self._encoder(input).permute(0, 2, 1)
        # quantized, indices, loss_vq = self._rvq(hidden)
        # # generate using conditioning
        # y_g_hat = self._generator(quantized.permute(0, 2, 1))
        hidden = self._encoder(input)  # .permute(0, 2, 1)
        # generate using conditioning
        y_g_hat = self._generator(hidden)  # .permute(0, 2, 1))
        return y_g_hat

    def training_step(self, batch, batch_ids):
        opt_g, opt_d = self.optimizers()
        x = batch['x']
        y = batch['y']
        sr = batch['sr']
        y = y.unsqueeze(1)
        denoise = batch['denoise']
        denoise = denoise.unsqueeze(1).repeat(1, 1, x.shape[1])
        x = x.unsqueeze(1)
        input = torch.cat([x, denoise], dim=1)
        # # compute conditioning using encoder and VQ
        # h = self._encoder(input).permute(0, 2, 1)
        #
        # quantized, indices, loss_vq = self._rvq(h)
        # # generate using conditioning
        # y_g_hat = self._generator(quantized.permute(0, 2, 1))
        h = self._encoder(input)  # .permute(0, 2, 1)

        y_g_hat = self._generator(h)
        m_size = min(y.shape[2], y_g_hat.shape[2])
        y = y[:, :, :m_size]
        y_g_hat = y_g_hat[:, :, :m_size]

        # select random section of audio (because we canot train the gan on the entire sequence)
        y_mel = mel_spectrogram(y.squeeze(1), 1024, 80, 48000, 240, 1024, 0, 24000)
        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), 1024, 80, 48000, 240, 1024, 0, 24000)

        opt_d.zero_grad()

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self._mpd(y, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self._msd(y, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f

        loss_disc_all.backward()
        opt_d.step()

        # Generator
        opt_g.zero_grad()

        # L1 Mel-Spectrogram Loss
        loss_mel = self._loss_l1(y_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self._mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self._msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel  # + loss_vq.sum()

        loss_gen_all.backward()
        opt_g.step()
        output_obj = {'loss_g': loss_gen_all,
                      # 'loss_vq': loss_vq.sum(),
                      'loss_d': loss_disc_all,
                      'loss_v': loss_gen_all + loss_disc_all,
                      'loss': loss_gen_all + loss_disc_all,
                      'lr': self._current_lr}
        self.log_dict(output_obj, prog_bar=True)
        self._global_step += 1
        self._current_lr = self._compute_lr(self._learning_rate, 1e-5, self._global_step)
        opt_d.param_groups[0]['lr'] = self._current_lr
        opt_g.param_groups[0]['lr'] = self._current_lr
        return output_obj

    def validation_step(self, batch, batch_ids):
        x = batch['x']
        y = batch['y']
        sr = batch['sr']
        y = y.unsqueeze(1)
        denoise = batch['denoise']
        denoise = denoise.unsqueeze(1).repeat(1, 1, x.shape[1])
        x = x.unsqueeze(1)
        input = torch.cat([x, denoise], dim=1)
        # # compute conditioning using encoder and VQ
        # h = self._encoder(input).permute(0, 2, 1)
        #
        # quantized, indices, loss_vq = self._rvq(h)
        # # generate using conditioning
        # y_g_hat = self._generator(quantized.permute(0, 2, 1))
        h = self._encoder(input)  # .permute(0, 2, 1)

        y_g_hat = self._generator(h)
        m_size = min(y.shape[2], y_g_hat.shape[2])
        y = y[:, :, :m_size]
        y_g_hat = y_g_hat[:, :, :m_size]

        # select random section of audio (because we canot train the gan on the entire sequence)
        y_mel = mel_spectrogram(y.squeeze(1), 1024, 80, 48000, 240, 1024, 0, 24000)
        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), 1024, 80, 48000, 240, 1024, 0, 24000)

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self._mpd(y, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self._msd(y, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f

        # L1 Mel-Spectrogram Loss
        loss_mel = self._loss_l1(y_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self._mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self._msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel  # + loss_vq.sum()

        output_obj = {'loss_g': loss_gen_all,
                      # 'loss_vq': loss_vq.sum(),
                      'loss_d': loss_disc_all,
                      'loss_v': loss_gen_all + loss_disc_all,
                      'loss': loss_gen_all + loss_disc_all,
                      'loss_mel': loss_mel,
                      'lr': self._current_lr}

        return output_obj

    def validation_epoch_end(self, outputs: []) -> None:
        target_loss = sum(x['loss_mel'] for x in outputs) / len(outputs)
        self._val_loss = target_loss

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(itertools.chain(self._generator.parameters(),
                                                    self._encoder.parameters(),
                                                    self._rvq.parameters()),
                                    self._current_lr, betas=[0.8, 0.99])
        optim_d = torch.optim.AdamW(itertools.chain(self._msd.parameters(),
                                                    self._mpd.parameters()
                                                    ),
                                    self._current_lr, betas=[0.8, 0.99])

        if self._loaded_optimizer_states is not None:
            for k, opt in zip(self._loaded_optimizer_states, [optim_g, optim_d]):
                if opt is not None:
                    opt_state = self._loaded_optimizer_states[k]
                    opt.load_state_dict(opt_state)
            self._loaded_optimizer_states = None  # free memory

        return optim_g, optim_d

    @torch.jit.ignore
    def save(self, path):
        torch.save(self.state_dict(), path)

    @torch.jit.ignore
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    @staticmethod
    def _compute_lr(initial_lr, delta, step):
        return initial_lr / (1 + delta * step)

    def get_device(self):
        if self._w.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._w.weight.device.type,
                                str(self._w.weight.device.index))
