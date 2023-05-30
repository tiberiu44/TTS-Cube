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
from cube.networks.modules import ConvNorm, LinearNorm, PreNet, PostNet, Languasito2
from collections import OrderedDict
from hifigan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from hifigan.env import AttrDict
from hifigan.meldataset import mel_spectrogram
from transformers import AutoModel


class Cubegan(pl.LightningModule):
    def __init__(self, encodings: CubeganEncodings, lr: float = 2e-4, conditioning=None, train=True):
        super(Cubegan, self).__init__()
        self._current_lr = lr
        self._learning_rate = lr
        self._global_step = 0
        self._encodings = encodings
        self._val_loss = 9999
        self._conditioning = conditioning
        self._loaded_optimizer_states = None
        if conditioning is not None:
            cond_type = conditioning.split(':')[0]
        else:
            cond_type = None
        self._cond_type = cond_type

        json_config = json.load(open('hifigan/config_v1.json'))
        h = AttrDict(json_config)
        self._generator = Generator(h)
        if train:
            self._mpd = MultiPeriodDiscriminator()
            self._msd = MultiScaleDiscriminator()
        self._languasito = Languasito2(len(encodings.phon2int), len(encodings.speaker2int), encodings.max_pitch,
                                       encodings.max_duration, cond_type=cond_type)
        self._loss_cross = nn.CrossEntropyLoss(ignore_index=int(max(encodings.max_pitch, encodings.max_duration) + 1))
        self._generator.train()
        if train:
            self._mpd.train()
            self._msd.train()

        if cond_type == 'hf':  # seems better to add the transformer model here
            self._hf = AutoModel.from_pretrained(conditioning.split(':')[-1])
        else:
            self._hf = None
            if train:
                self._dummy = nn.Linear(1, 1)

        self._loss_l1 = nn.L1Loss()
        self.automatic_optimization = False

    def forward(self, X):
        with torch.no_grad():
            if self._hf is not None:
                hf_cond = self._hf(X['x_tok_ids'])['last_hidden_state']
            else:
                hf_cond = None
            _, _, _, conditioning = self._languasito(X, hf_cond=hf_cond)
            return self._generator(conditioning.permute(0, 2, 1))

    def inference(self, X):
        with torch.no_grad():
            if self._hf is not None:
                hf_cond = self._hf(X['x_tok_ids'])['last_hidden_state']
            else:
                hf_cond = None
            conditioning = self._languasito.inference(X, hf_cond=hf_cond)
            if conditioning.shape[1] == 0:
                conditioning = torch.zeros((conditioning.shape[0], 1, conditioning.shape[2]), device=self.get_device())
            return self._generator(conditioning.permute(0, 2, 1))

    def training_step(self, batch, batch_ids):
        opt_g, opt_d, opt_t, opt_b = self.optimizers()

        if self._hf is not None:
            hf_cond = self._hf(batch['x_tok_ids'])['last_hidden_state']
        else:
            hf_cond = None

        p_dur, p_pitch, p_vuv, conditioning = self._languasito(batch, hf_cond=hf_cond)
        t_dur = batch['y_dur']
        t_pitch = batch['y_pitch']
        t_vuv = (t_pitch > 1).float()
        # match shapes
        m_size = min(t_dur.shape[1], p_dur.shape[1])
        t_dur = t_dur[:, :m_size]
        p_dur = p_dur[:, :m_size, :]
        m_size = min(t_pitch.shape[1], p_pitch.shape[1])
        t_pitch = t_pitch[:, :m_size]
        p_pitch = p_pitch[:, :m_size]
        t_vuv = t_vuv[:, :m_size]
        p_vuv = p_vuv[:, :m_size]

        loss_duration = self._loss_cross(p_dur.reshape(-1, p_dur.shape[2]), t_dur.reshape(-1))

        # loss_pitch = self._loss_cross(p_pitch.reshape(-1, p_pitch.shape[2]),
        #                               t_pitch.reshape(-1))
        loss_pitch = (torch.abs(t_pitch / self._languasito._max_pitch - p_pitch) * t_vuv).mean() + \
                     torch.abs(t_vuv - p_vuv).mean()

        y = batch['y_audio']

        if y.shape[1] > 12000 - 240:
            y_t_list = []
            c_list = []
            for ii in range(y.shape[0]):
                max_frame = len(batch['y_frame2phone'][ii])
                if max_frame > 51:
                    r = random.randint(0, max_frame - 50 - 1)
                else:
                    r = 0
                c_list.append(conditioning[ii, r:r + 50, :].unsqueeze(0))
                y_t_list.append(y[ii, r * 240:r * 240 + 12000].unsqueeze(0))
            y = torch.cat(y_t_list, dim=0)
            conditioning = torch.cat(c_list, dim=0)

        y = y.unsqueeze(1)
        y_g_hat = self._generator(conditioning.permute(0, 2, 1))
        m_size = min(y.shape[2], y_g_hat.shape[2])
        y = y[:, :, :m_size]
        y_g_hat = y_g_hat[:, :, :m_size]

        # select random section of audio (because we canot train the gan on the entire sequence)
        y_mel = mel_spectrogram(y.squeeze(1), 1024, 80, 24000, 240, 1024, 0, 12000)
        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), 1024, 80, 24000, 240, 1024, 0, 12000)

        opt_b.zero_grad()
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
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        loss_gen_all.backward(retain_graph=True)
        opt_g.step()
        opt_t.zero_grad()
        loss_text = loss_pitch + loss_duration
        loss_text.backward(retain_graph=True)
        opt_t.step()
        opt_b.step()
        output_obj = {'loss_g': loss_gen_all,
                      'loss_t': loss_text,
                      'loss_d': loss_disc_all,
                      'loss_v': loss_gen_all + loss_disc_all,
                      'loss': loss_gen_all + loss_disc_all + loss_text,
                      'lr': self._current_lr}
        self.log_dict(output_obj, prog_bar=True)
        self._global_step += 1
        self._current_lr = self._compute_lr(self._learning_rate, 1e-5, self._global_step)
        opt_d.param_groups[0]['lr'] = self._current_lr
        opt_g.param_groups[0]['lr'] = self._current_lr
        opt_t.param_groups[0]['lr'] = self._current_lr
        return output_obj

    def validation_step(self, batch, batch_ids):
        if self._hf is not None:
            hf_cond = self._hf(batch['x_tok_ids'])['last_hidden_state']
        else:
            hf_cond = None
        p_dur, p_pitch, p_vuv, conditioning = self._languasito(batch, hf_cond=hf_cond)

        t_dur = batch['y_dur']
        t_pitch = batch['y_pitch']
        t_vuv = (t_pitch > 1).float()
        # match shapes
        m_size = min(t_dur.shape[1], p_dur.shape[1])
        t_dur = t_dur[:, :m_size]
        p_dur = p_dur[:, :m_size, :]
        m_size = min(t_pitch.shape[1], p_pitch.shape[1])
        t_pitch = t_pitch[:, :m_size]
        p_pitch = p_pitch[:, :m_size]
        t_vuv = t_vuv[:, :m_size]
        p_vuv = p_vuv[:, :m_size]

        loss_duration = self._loss_cross(p_dur.reshape(-1, p_dur.shape[2]), t_dur.reshape(-1))

        loss_pitch = (torch.abs(t_pitch / self._languasito._max_pitch - p_pitch) * t_vuv).mean() + \
                     torch.abs(t_vuv - p_vuv).mean()

        y = batch['y_audio']
        # select random section of audio (because we canot train the gan on the entire sequence
        if y.shape[1] > 48000 - 240:
            y_t_list = []
            c_list = []
            for ii in range(y.shape[0]):
                max_frame = len(batch['y_frame2phone'][ii])
                if max_frame > 201:
                    r = random.randint(0, max_frame - 200 - 1)
                else:
                    r = 0
                c_list.append(conditioning[ii, r:r + 200, :].unsqueeze(0))
                y_t_list.append(y[ii, r * 240:r * 240 + 48000].unsqueeze(0))
            y = torch.cat(y_t_list, dim=0)
            conditioning = torch.cat(c_list, dim=0)

        y = y.unsqueeze(1)

        y_g_hat = self._generator(conditioning.permute(0, 2, 1))
        m_size = min(y.shape[2], y_g_hat.shape[2])
        y = y[:, :, :m_size]
        y_g_hat = y_g_hat[:, :, :m_size]

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
        loss_mel = self._loss_l1(y_mel, y_g_hat_mel)

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self._mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self._msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
        loss_text = loss_pitch + loss_duration

        return {'loss_g': loss_gen_all,
                'loss_d': loss_disc_all,
                'loss': loss_gen_all + loss_disc_all + loss_text,
                'loss_v': loss_gen_all + loss_disc_all,
                'loss_mel': loss_mel}

    def validation_epoch_end(self, outputs: []) -> None:
        target_loss = sum(x['loss_mel'] for x in outputs) / len(outputs)
        self._val_loss = target_loss

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(itertools.chain(self._generator.parameters(),
                                                    self._languasito._phon_emb_g.parameters(),
                                                    self._languasito._speaker_emb_g.parameters(),
                                                    self._languasito._char_cnn_g.parameters(),
                                                    self._languasito._char_rnn_g.parameters(),
                                                    self._languasito._lm_g.parameters(),
                                                    self._languasito._cond_rnn.parameters(),
                                                    self._languasito._cond_output.parameters()),
                                    self._current_lr, betas=[0.8, 0.99])
        optim_d = torch.optim.AdamW(itertools.chain(self._msd.parameters(),
                                                    self._mpd.parameters()
                                                    ),
                                    self._current_lr, betas=[0.8, 0.99])
        optim_t = torch.optim.AdamW(itertools.chain(self._languasito._phon_emb_t.parameters(),
                                                    self._languasito._speaker_emb_t.parameters(),
                                                    self._languasito._char_cnn_t.parameters(),
                                                    self._languasito._char_rnn_t.parameters(),
                                                    self._languasito._lm_t.parameters(),
                                                    self._languasito._dur_rnn.parameters(),
                                                    self._languasito._dur_output.parameters(),
                                                    self._languasito._pitch_rnn.parameters(),
                                                    self._languasito._pitch_output.parameters()),
                                    self._current_lr, betas=[0.8, 0.99])
        if self._cond_type == 'hf':
            optim_b = torch.optim.Adam(self._hf.parameters(), lr=1e-6)
        else:
            optim_b = torch.optim.Adam(self._dummy.parameters(), lr=1e-6)

        if self._loaded_optimizer_states is not None:
            for k, opt in zip(self._loaded_optimizer_states, [optim_g, optim_d, optim_t, optim_b]):
                if opt is not None:
                    opt_state = self._loaded_optimizer_states[k]
                    opt.load_state_dict(opt_state)
            self._loaded_optimizer_states = None  # free memory

        return optim_g, optim_d, optim_t, optim_b

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
