import random

import torch
import torch.nn as nn

import torch
import json
import torch.nn as nn
import pytorch_lightning as pl
import sys
import itertools

sys.path.append('')
sys.path.append('hifigan')
from cube.io_utils.io_cubegan import CubeganEncodings
from cube.networks.modules import ConvNorm, LinearNorm, PreNet, PostNet, Languasito
from collections import OrderedDict
from hifigan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from hifigan.env import AttrDict
from hifigan.meldataset import mel_spectrogram


class Cubegan(pl.LightningModule):
    def __init__(self, encodings: CubeganEncodings, lr: float = 2e-4):
        super(Cubegan, self).__init__()
        self._lr = lr
        self._encodings = encodings
        self._val_loss = 9999

        json_config = json.load(open('hifigan/config_v1.json'))
        h = AttrDict(json_config)
        self._generator = Generator(h)
        self._mpd = MultiPeriodDiscriminator()
        self._msd = MultiScaleDiscriminator()
        self._languasito = Languasito(len(encodings.phon2int), len(encodings.speaker2int), encodings.max_pitch,
                                      encodings.max_duration)
        self._loss_cross = nn.CrossEntropyLoss(ignore_index=int(max(encodings.max_pitch, encodings.max_duration) + 1))
        self._generator.train()
        self._mpd.train()
        self._msd.train()

        self._loss_l1 = nn.L1Loss()
        self.automatic_optimization = False

    def forward(self, X):
        pass

    def inference(self, X):
        with torch.no_grad():
            conditioning = self._languasito.inference(X)
            return self._generator(conditioning.permute(0, 2, 1))

    def training_step(self, batch, batch_ids):
        opt_g, opt_d = self.optimizers()

        p_dur, p_pitch, conditioning = self._languasito(batch)
        t_dur = batch['y_dur']
        t_pitch = batch['y_pitch']
        # match shapes
        m_size = min(t_dur.shape[1], p_dur.shape[1])
        t_dur = t_dur[:, :m_size]
        p_dur = p_dur[:, :m_size, :]
        m_size = min(t_pitch.shape[1], p_pitch.shape[1])
        t_pitch = t_pitch[:, :m_size]
        p_pitch = p_pitch[:, :m_size, :]

        loss_duration = self._loss_cross(p_dur.reshape(-1, p_dur.shape[2]), t_dur.reshape(-1))

        loss_pitch = self._loss_cross(p_pitch.reshape(-1, p_pitch.shape[2]),
                                      t_pitch.reshape(-1))

        y = batch['y_audio'].unsqueeze(1)
        y_g_hat = self._generator(conditioning.permute(0, 2, 1))
        m_size = min(y.shape[2], y_g_hat.shape[2])
        y = y[:, :, :m_size]
        y_g_hat = y_g_hat[:, :, :m_size]

        # select random section of audio (because we canot train the gan on the entire sequence)
        if y.shape[2] > 48000:
            r = random.randint(0, m_size - 1 - 48000)
            y = y[:, :, r:r + 48000]
            y_g_hat = y_g_hat[:, :, r:r + 48000]
        y_mel = mel_spectrogram(y.squeeze(1), 1024, 80, 24000, 240, 1024, 0, 12000)
        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), 1024, 80, 24000, 240, 1024, 0, 12000)

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
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_pitch + loss_duration

        loss_gen_all.backward()
        opt_g.step()
        output_obj = {'loss_g': loss_gen_all, 'loss_d': loss_disc_all, 'loss': loss_gen_all + loss_disc_all}
        self.log_dict(output_obj, prog_bar=True)
        return output_obj

    def validation_step(self, batch, batch_ids):
        p_dur, p_pitch, conditioning = self._languasito(batch)
        t_dur = batch['y_dur']
        t_pitch = batch['y_pitch']
        # match shapes
        m_size = min(t_dur.shape[1], p_dur.shape[1])
        t_dur = t_dur[:, :m_size]
        p_dur = p_dur[:, :m_size, :]
        m_size = min(t_pitch.shape[1], p_pitch.shape[1])
        t_pitch = t_pitch[:, :m_size]
        p_pitch = p_pitch[:, :m_size, :]

        loss_duration = self._loss_cross(p_dur.reshape(-1, p_dur.shape[2]), t_dur.reshape(-1))

        loss_pitch = self._loss_cross(p_pitch.reshape(-1, p_pitch.shape[2]),
                                      t_pitch.reshape(-1))

        y = batch['y_audio'].unsqueeze(1)

        if y.shape[2] > 48000:
            r = random.randint(0, m_size - 1 - 48000) // 240 * 240
            y = y[:, :, r:r + 48000]
            conditioning = conditioning[:, r // 240:r // 240 + 200]
            
        y_g_hat = self._generator(conditioning.permute(0, 2, 1))
        m_size = min(y.shape[2], y_g_hat.shape[2])
        y = y[:, :, :m_size]
        y_g_hat = y_g_hat[:, :, :m_size]

        # select random section of audio (because we canot train the gan on the entire sequence)
        if y.shape[2] > 12000:
            r = random.randint(0, m_size - 1 - 12000)
            y = y[:, :, r:r + 12000]
            y_g_hat = y_g_hat[:, :, r:r + 12000]
        y_mel = mel_spectrogram(y.squeeze(1), 1024, 80, 24000, 240, 1024, 0, 12000)
        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), 1024, 80, 24000, 240, 1024, 0, 12000)

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self._mpd(y, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self._msd(y, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f

        # Generator
        # L1 Mel-Spectrogram Loss
        loss_mel = self._loss_l1(y_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self._mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self._msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_pitch + loss_duration

        return {'loss_g': loss_gen_all, 'loss_d': loss_disc_all, 'loss': loss_gen_all + loss_disc_all}

    def validation_epoch_end(self, outputs: []) -> None:
        total_loss = sum(x['loss'] for x in outputs) / len(outputs)
        self._val_loss = total_loss

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(itertools.chain(self._generator.parameters(),
                                                    self._languasito.parameters()),
                                    self._lr, betas=[0.8, 0.99])
        optim_d = torch.optim.AdamW(itertools.chain(self._msd.parameters(),
                                                    self._mpd.parameters()
                                                    ),
                                    self._lr, betas=[0.8, 0.99])
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
        return self._languasito._get_device()
