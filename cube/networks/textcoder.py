import torch
import torch.nn as nn
import pytorch_lightning as pl
import sys

sys.path.append('')
from cube.io_utils.io_textcoder import TextcoderEncodings
from cube.networks.modules import ConvNorm, LinearNorm, PreNet, PostNet
from collections import OrderedDict


class CubenetTextcoder(pl.LightningModule):
    def __init__(self, encodings: TextcoderEncodings, pframes: int = 3, lr: float = 2e-4):
        super(CubenetTextcoder, self).__init__()
        PHON_EMB_SIZE = 64
        SPEAKER_EMB_SIZE = 128
        CHAR_CNN_SIZE = 256
        CHAR_CNN_KS = 3
        CHAR_CNN_NL = 3
        CHAR_RNN_NL = 2
        CHAR_RNN_SIZE = 256
        OVERLAY_RNN_LAYERS = 2
        OVERLAY_RNN_SIZE = 512
        EXTERNAL_COND = 0  # this will be used to add external conditioning (e.g. transformer) - not currently used
        DUR_RNN_SIZE = 256
        DUR_RNN_LAYERS = 2
        PITCH_RNN_SIZE = 256
        PITCH_RNN_LAYERS = 2
        MEL_RNN_SIZE = 512
        MEL_RNN_LAYERS = 2
        PRENET_SIZE = 256
        PRENET_LAYERS = 2
        MEL_SIZE = 80
        self._pframes = pframes
        self._lr = lr
        self._encodings = encodings
        # phoneme embeddings
        self._phon_emb = nn.Embedding(len(encodings.phon2int) + 1, PHON_EMB_SIZE, padding_idx=0)
        # speaker embeddings
        self._speaker_emb = nn.Embedding(len(encodings.speaker2int) + 1, SPEAKER_EMB_SIZE, padding_idx=0)
        # phoneme/char CNN
        inp_s = PHON_EMB_SIZE
        char_cnn = []
        for ii in range(CHAR_CNN_NL):
            conv = ConvNorm(inp_s,
                            CHAR_CNN_SIZE,
                            kernel_size=CHAR_CNN_KS,
                            padding=CHAR_CNN_KS // 2,
                            w_init_gain='tanh')
            char_cnn.append(conv)
            char_cnn.append(nn.Tanh())
            inp_s = CHAR_CNN_SIZE
        self._char_cnn = nn.ModuleList(char_cnn)
        # phoneme/char RNN
        self._rnn_char = nn.LSTM(input_size=inp_s,
                                 hidden_size=CHAR_RNN_SIZE,
                                 num_layers=CHAR_RNN_NL,
                                 bidirectional=True,
                                 batch_first=True)
        # rnn over the upsampled data - this helps with co-articulation of sounds
        self._rnn_overlay = nn.LSTM(input_size=CHAR_RNN_SIZE * 2 + SPEAKER_EMB_SIZE + EXTERNAL_COND,
                                    hidden_size=OVERLAY_RNN_SIZE,
                                    num_layers=OVERLAY_RNN_LAYERS,
                                    bidirectional=True,
                                    batch_first=True)
        # duration
        # this comes after the textcnn+speaker_emb+external_cond(could be a transformer)
        self._dur_rnn = nn.LSTM(input_size=CHAR_RNN_SIZE * 2 + SPEAKER_EMB_SIZE + EXTERNAL_COND,
                                hidden_size=DUR_RNN_SIZE,
                                num_layers=DUR_RNN_LAYERS,
                                bidirectional=True,
                                batch_first=True)
        self._dur_output = LinearNorm(DUR_RNN_SIZE * 2, encodings.max_duration + 1)
        # pitch
        # this comes after the rnn_overlay+speaker_emb+external_cond(could be a transformer)
        self._pitch_rnn = nn.LSTM(input_size=OVERLAY_RNN_SIZE * 2,
                                  hidden_size=PITCH_RNN_SIZE,
                                  num_layers=PITCH_RNN_LAYERS,
                                  bidirectional=True,
                                  batch_first=True)
        self._pitch_output = LinearNorm(PITCH_RNN_SIZE * 2, int(encodings.max_pitch) + 1)
        # mel
        # this comes after the rnn_overlay
        self._mel_rnn = nn.LSTM(input_size=OVERLAY_RNN_SIZE * 2 + PRENET_SIZE,
                                hidden_size=MEL_RNN_SIZE,
                                num_layers=MEL_RNN_LAYERS,
                                bidirectional=False,
                                batch_first=True)
        self._mel_output = LinearNorm(MEL_RNN_SIZE, MEL_SIZE * pframes)
        self._prenet = PreNet(MEL_SIZE, PRENET_SIZE, PRENET_LAYERS)
        self._postnet = PostNet(MEL_SIZE)
        self.automatic_optimization = False
        self._loss_l1 = nn.L1Loss()
        self._loss_mse = nn.MSELoss()
        self._loss_cross = nn.CrossEntropyLoss(ignore_index=int(max(encodings.max_pitch, encodings.max_duration) + 1))
        self._val_loss_durs = 9999
        self._val_loss_pitch = 9999
        self._val_loss_mel = 9999
        self._val_loss_total = 999

    def forward(self, X):
        x_char = X['x_char']
        x_speaker = X['x_speaker']
        speaker_cond = self._speaker_emb(x_speaker)
        # compute character embeddings
        hidden = self._phon_emb(x_char)
        hidden = hidden.permute(0, 2, 1)
        for layer in self._char_cnn:
            hidden = layer(hidden)
        hidden = hidden.permute(0, 2, 1)
        hidden, _ = self._rnn_char(hidden)
        # done with character processing
        # append speaker and, if possible, external cond
        expanded_speaker = speaker_cond.repeat(1, hidden.shape[1], 1)
        hidden = torch.cat([hidden, expanded_speaker], dim=-1)
        # duration
        hidden_dur, _ = self._dur_rnn(hidden)
        output_dur = self._dur_output(hidden_dur)

        # align/repeat to match alignments
        hidden = self._expand(hidden, X['y_frame2phone'])
        # overlay
        hidden, _ = self._rnn_overlay(hidden)
        # pitch
        hidden_pitch, _ = self._pitch_rnn(hidden)
        output_pitch = self._pitch_output(hidden_pitch)
        # mel
        cond_mel = self._prepare_mel(X['y_mgc'])
        cond_mel = self._prenet(cond_mel)
        m_size = min(hidden.shape[1], cond_mel.shape[1])
        hidden = torch.cat([hidden[:, :m_size, :], cond_mel[:, :m_size, :]], dim=-1)
        hidden_mel, _ = self._mel_rnn(hidden)
        output_mel = self._mel_output(hidden_mel)
        output_mel = output_mel.reshape(output_mel.shape[0], -1, 80)
        post_out = self._postnet(output_mel)
        output_mel_post = output_mel + post_out

        return output_dur, output_pitch, output_mel, output_mel_post

    def inference(self, X):
        x_char = X['x_char']
        x_speaker = X['x_speaker']
        speaker_cond = self._speaker_emb(x_speaker)
        # compute character embeddings
        hidden = self._phon_emb(x_char)
        hidden = hidden.permute(0, 2, 1)
        for layer in self._char_cnn:
            hidden = layer(hidden)
        hidden = hidden.permute(0, 2, 1)
        hidden, _ = self._rnn_char(hidden)
        # done with character processing
        # append speaker and, if possible, external cond
        expanded_speaker = speaker_cond.repeat(1, hidden.shape[1], 1)
        hidden = torch.cat([hidden, expanded_speaker], dim=-1)
        # duration
        hidden_dur, _ = self._dur_rnn(hidden)
        output_dur = self._dur_output(hidden_dur)
        # we have the durations, we need to simulate alignments here
        output_dur = torch.argmax(output_dur, dim=-1)
        frame2phone = []
        phon_index = 0
        for dur in output_dur.detach().cpu().numpy().squeeze():
            for ii in range(dur):
                frame2phone.append(phon_index)
            phon_index += 1
        # align/repeat to match alignments
        hidden = self._expand(hidden, [frame2phone])
        # overlay
        hidden, _ = self._rnn_overlay(hidden)
        # # pitch is not currently used by the model
        # hidden_pitch, _ = self._pitch_rnn(hidden)
        # output_pitch = self._pitch_output(hidden_pitch)
        # mel
        last_mel = torch.ones((hidden.shape[0], 1, 80), device=self._get_device()) * -5
        hx = None
        output_mel_list = []
        for ii in range(hidden.shape[1]):
            last_mel = self._prenet(last_mel)
            hid_tmp = torch.cat([hidden[:, ii, :].unsqueeze(1), last_mel], dim=-1)
            hidden_mel, hx = self._mel_rnn(hid_tmp, hx=hx)
            output_mel = self._mel_output(hidden_mel)
            output_mel_list.append(output_mel)
            last_mel = output_mel[:, :, -80:]
        output_mel = torch.cat(output_mel_list, dim=1)
        output_mel = output_mel.reshape(output_mel.shape[0], -1, 80)
        post_out = self._postnet(output_mel)
        output_mel_post = output_mel + post_out

        return output_mel_post

    def training_step(self, batch, batch_ids):
        opt = self.optimizers()
        opt.zero_grad()
        p_dur, p_pitch, pre_mel, post_mel = self.forward(batch)
        t_dur = batch['y_dur']
        t_pitch = self._prepare_pitch(batch['y_pitch'])
        t_mel = batch['y_mgc']
        # match shapes
        m_size = min(t_dur.shape[1], p_dur.shape[1])
        t_dur = t_dur[:, :m_size]
        p_dur = p_dur[:, :m_size, :]
        m_size = min(t_pitch.shape[1], p_pitch.shape[1])
        t_pitch = t_pitch[:, :m_size]
        p_pitch = p_pitch[:, :m_size, :]
        m_size = min(pre_mel.shape[1], t_mel.shape[1])
        pre_mel = pre_mel[:, :m_size, :]
        post_mel = post_mel[:, :m_size, :]
        t_mel = t_mel[:, :m_size, :]
        loss_duration = self._loss_cross(p_dur.reshape(-1, p_dur.shape[2]), t_dur.reshape(-1))

        loss_pitch = self._loss_cross(p_pitch.reshape(-1, p_pitch.shape[2]),
                                      t_pitch.reshape(-1))
        loss_mel = self._loss_l1(pre_mel, t_mel) + \
                   self._loss_l1(post_mel, t_mel)
        loss = loss_duration + loss_pitch + loss_mel
        loss.backward()
        opt.step()
        output_obj = {
            'loss': loss,
            'l_mel': loss_mel,
            'l_pitch': loss_pitch,
            'l_dur': loss_duration
        }
        self.log_dict(output_obj, prog_bar=True)

        return output_obj

    def validation_step(self, batch, batch_ids):
        p_dur, p_pitch, pre_mel, post_mel = self.forward(batch)
        t_dur = batch['y_dur']
        t_pitch = self._prepare_pitch(batch['y_pitch'])
        t_mel = batch['y_mgc']
        # match shapes
        m_size = min(t_dur.shape[1], p_dur.shape[1])
        t_dur = t_dur[:, :m_size]
        p_dur = p_dur[:, :m_size, :]
        m_size = min(t_pitch.shape[1], p_pitch.shape[1])
        t_pitch = t_pitch[:, :m_size]
        p_pitch = p_pitch[:, :m_size, :]
        m_size = min(pre_mel.shape[1], t_mel.shape[1])
        pre_mel = pre_mel[:, :m_size, :]
        post_mel = post_mel[:, :m_size, :]
        t_mel = t_mel[:, :m_size, :]
        loss_duration = self._loss_cross(p_dur.reshape(-1, p_dur.shape[2]), t_dur.reshape(-1))

        loss_pitch = self._loss_cross(p_pitch.reshape(-1, p_pitch.shape[2]),
                                      t_pitch.reshape(-1))
        loss_mel = self._loss_l1(pre_mel, t_mel) + \
                   self._loss_l1(post_mel, t_mel)
        output_obj = {
            'loss': loss_duration + loss_pitch + loss_mel,
            'l_mel': loss_mel,
            'l_pitch': loss_pitch,
            'l_dur': loss_duration
        }
        self.log_dict(output_obj, prog_bar=True)
        return output_obj

    def validation_epoch_end(self, outputs: []) -> None:
        loss_total = sum([output['loss'] for output in outputs]) / len(outputs)
        loss_mel = sum([output['l_mel'] for output in outputs]) / len(outputs)
        loss_pitch = sum([output['l_pitch'] for output in outputs]) / len(outputs)
        loss_dur = sum([output['l_dur'] for output in outputs]) / len(outputs)
        self._val_loss_durs = loss_dur.item()
        self._val_loss_pitch = loss_pitch.item()
        self._val_loss_mel = loss_mel.item()
        self._val_loss_total = loss_total.item()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)

    @torch.jit.ignore
    def save(self, path):
        torch.save(self.state_dict(), path)

    @torch.jit.ignore
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    @staticmethod
    def _compute_lr(initial_lr, delta, step):
        return initial_lr / (1 + delta * step)

    @torch.jit.ignore
    def _get_device(self):
        if self._mel_output.linear_layer.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._mel_output.linear_layer.weight.device.type,
                                str(self._mel_output.linear_layer.weight.device.index))

    def _expand(self, x, alignments):
        m_size = max([len(a) // self._pframes for a in alignments])
        tmp = []
        for ii in range(len(alignments)):
            c_batch = []
            for jj in range(len(alignments[ii]) // self._pframes):
                c_batch.append(x[ii, alignments[ii][jj * self._pframes], :].unsqueeze(0))
            for jj in range(m_size - len(alignments[ii]) // self._pframes):
                c_batch.append(x[ii, -1, :].unsqueeze(0))
            c_batch = torch.cat(c_batch, dim=0)
            tmp.append(c_batch.unsqueeze(0))
        return torch.cat(tmp, dim=0)

    def _prepare_mel(self, x):
        lst = [torch.ones((x.shape[0], 1, x.shape[2]), device=self._get_device()) * -5]
        for ii in range(x.shape[1] // self._pframes):
            lst.append(x[:, (ii + 1) * self._pframes - 1, :].unsqueeze(1))
        return torch.cat(lst, dim=1)

    def _prepare_pitch(self, x):
        lst = []
        for ii in range(x.shape[1] // self._pframes):
            lst.append(x[:, (ii + 1) * self._pframes - 1].unsqueeze(1))
        return torch.cat(lst, dim=1)
