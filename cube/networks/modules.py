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
import torch.nn as nn
import numpy as np
import tqdm
from cube.networks.loss import BetaOutput, GaussianOutput, MULAWOutput, MOLOutput, RAWOutput


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_normal_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_normal_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, att_proj_size=100, dropout_prob=0.1, kernel_size=1):
        super(Attention, self).__init__()

        self.dropout_prob = dropout_prob
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = ConvNorm(enc_hid_dim + dec_hid_dim, att_proj_size, kernel_size=kernel_size,
                             w_init_gain='tanh')  # LinearNorm((enc_hid_dim * 2) + dec_hid_dim, att_proj_size, w_init_gain='tanh')
        self.v = nn.Parameter(torch.rand(att_proj_size))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.dropout(
            torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2).permute(0, 2, 1)).permute(0, 2, 1)),
            self.dropout_prob,
            self.training)
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.softmax(torch.bmm(v, energy).squeeze(1), dim=1)

        a = attention.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.squeeze(1)

        return attention, weighted


class ResNet(nn.Module):
    def __init__(self, input_size, filter_size=512, n_layers=8, kernel_size=3):
        super(ResNet, self).__init__()
        conv_list = []
        i_size = input_size
        for ii in range(n_layers):
            conv = ConvNorm(i_size, filter_size, kernel_size, padding=kernel_size // 2)
            i_size = filter_size // 2
            conv_list.append(conv)
        self._convs = nn.ModuleList(conv_list)
        self._half = filter_size // 2

    def forward(self, x):
        h = x.permute(0, 2, 1)
        res = None
        for conv in self._convs:
            output_h = conv(h)
            gate = torch.sigmoid(output_h[:, :self._half, :])
            act = torch.tanh(output_h[:, self._half:, :])
            h = gate * act
            if res is not None:
                h = h + res
            res = h
        return h.permute(0, 2, 1)


class PostNet(nn.Module):
    def __init__(self, num_mels=80, kernel_size=5, filter_size=512, output_size=None):
        super(PostNet, self).__init__()
        if output_size is None:
            output_size = num_mels
        self.network = nn.Sequential(
            ConvNorm(num_mels, filter_size, kernel_size, padding=kernel_size // 2, w_init_gain='tanh'),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.1),
            ConvNorm(filter_size, filter_size, kernel_size, padding=kernel_size // 2, w_init_gain='tanh'),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.1),
            ConvNorm(filter_size, filter_size, kernel_size, padding=kernel_size // 2, w_init_gain='tanh'),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.1),
            ConvNorm(filter_size, filter_size, kernel_size, padding=kernel_size // 2, w_init_gain='tanh'),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.1),
            ConvNorm(filter_size, output_size, kernel_size, padding=kernel_size // 2, w_init_gain='linear'),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.network(x).permute(0, 2, 1)
        return y


class PreNet(nn.Module):
    def __init__(self, num_mels=80, hidden=256, layers=2):
        super(PreNet, self).__init__()
        module_list = []
        inp_size = num_mels
        for ii in range(layers):
            module_list.append(LinearNorm(inp_size, hidden, w_init_gain='linear'))
            inp_size = hidden

        self.layers_h = nn.ModuleList(module_list)

    def forward(self, x):
        hidden = x
        for layer in self.layers_h:
            hidden = torch.relu(layer(hidden))
            hidden = torch.dropout(hidden, 0.5, True)
        return hidden


class Mel2Style(nn.Module):
    def __init__(self, num_mgc=80, gst_dim=100, num_gst=8, rnn_size=128, rnn_layers=1):
        super(Mel2Style, self).__init__()
        self.dec_hid_dim = rnn_size
        self.num_gst = num_gst

        self.attn = LinearNorm(gst_dim + rnn_size, rnn_size, w_init_gain='tanh')
        self.v = nn.Parameter(torch.rand(rnn_size))
        self.lstm = nn.LSTM(num_mgc, rnn_size, rnn_layers, batch_first=True)

    def forward(self, mgc, gst):
        # invert sequence - no pytorch function found
        mgc_list = []
        for ii in range(mgc.shape[1]):
            mgc_list.append(mgc[:, mgc.shape[1] - ii - 1, :].unsqueeze(1))
        # from ipdb import set_trace
        # set_trace()
        mgc = torch.cat(mgc_list, dim=1)
        hidden, _ = self.lstm(mgc)
        hidden = hidden[:, -1, :]
        batch_size = hidden.shape[0]
        src_len = self.num_gst
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        unfolded_gst = torch.tensor([[i for i in range(self.num_gst)] for _ in range(batch_size)],
                                    device=self._get_device(), dtype=torch.long)
        encoder_outputs = torch.tanh(gst(unfolded_gst))
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.softmax(torch.bmm(v, energy).squeeze(1), dim=1)
        a = attention.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs).squeeze(1)
        return attention, weighted

    def _get_device(self):
        if self.attn.linear_layer.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self.attn.linear_layer.weight.device.type,
                                str(self.attn.linear_layer.weight.device.index))


class Seq2Seq(nn.Module):
    def __init__(self, num_input_tokens, num_output_tokens, embedding_size=100, encoder_size=100, encoder_layers=2,
                 decoder_size=200, decoder_layers=2, pad_index=0, unk_index=1, stop_index=2):
        super(Seq2Seq, self).__init__()
        self.emb_size = embedding_size
        self.input_emb = nn.Embedding(num_input_tokens, embedding_size, padding_idx=pad_index)
        self.output_emb = nn.Embedding(num_output_tokens, embedding_size, padding_idx=pad_index)
        self.encoder = nn.LSTM(embedding_size, encoder_size, encoder_layers, dropout=0.33, bidirectional=True,
                               batch_first=True)
        self.decoder = nn.LSTM(encoder_size * 2 + embedding_size, decoder_size, decoder_layers, dropout=0.33,
                               batch_first=True)
        self.attention = Attention(encoder_size * 2, decoder_size, att_proj_size=decoder_size)
        self.output = nn.Linear(decoder_size, num_output_tokens)
        self._PAD = pad_index
        self._UNK = unk_index
        self._EOS = stop_index
        self._dec_input_size = encoder_size * 2 + embedding_size

    def inference(self, x):
        x = self.input_emb(x)
        encoder_output, encoder_hidden = self.encoder(x)
        # encoder_output = encoder_output.permute(1, 0, 2)
        count = 0

        _, decoder_hidden = self.decoder(torch.zeros((x.shape[0], 1, self._dec_input_size)))
        last_output_emb = torch.zeros((x.shape[0], self.emb_size))
        output_list = []
        index = 0
        reached_end = [False for ii in range(x.shape[0])]
        while True:
            _, encoder_att = self.attention(decoder_hidden[-1][-1], encoder_output)
            decoder_input = torch.cat([encoder_att, last_output_emb], dim=1)
            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
            output = self.output(decoder_output.squeeze(1))
            output_list.append(output.unsqueeze(1))

            outp = torch.argmax(output, dim=1)
            last_output_emb = self.output_emb(outp)
            for ii in range(outp.shape[0]):
                if outp[ii] == self._EOS:
                    reached_end[ii] = True
            if np.all(reached_end):
                break
            index += 1

            if index > x.shape[1] * 10:
                break

        return torch.cat(output_list, dim=1)

    def forward(self, x, gs_output=None):
        # x, y = self._make_batches(input, gs_output)
        x = self.input_emb(x)
        encoder_output, encoder_hidden = self.encoder(x)
        count = 0
        if gs_output is not None:
            batch_output_emb = self.output_emb(gs_output)

        _, decoder_hidden = self.decoder(torch.zeros((x.shape[0], 1, self._dec_input_size), device=self._get_device()))
        last_output_emb = torch.zeros((x.shape[0], self.emb_size), device=self._get_device())
        output_list = []
        index = 0
        reached_end = [False for _ in range(x.shape[0])]
        while True:
            _, encoder_att = self.attention(decoder_hidden[-1][-1], encoder_output)
            decoder_input = torch.cat([encoder_att, last_output_emb], dim=1)
            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
            output = self.output(decoder_output.squeeze(1))
            output_list.append(output.unsqueeze(1))

            if gs_output is not None:
                last_output_emb = batch_output_emb[:, index, :]
                index += 1
                if index == gs_output.shape[1]:
                    break
            else:
                outp = torch.argmax(output, dim=1)
                last_output_emb = self.output_emb(outp)
                for ii in range(outp.shape[0]):
                    if outp[ii] == self._EOS:
                        reached_end[ii] = True

                if np.all(reached_end):
                    break
                index += 1

                if index > x.shape[1] * 10:
                    break

        return torch.cat(output_list, dim=1)

    @torch.jit.ignore
    def _get_device(self):
        if self.input_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self.input_emb.weight.device.type, str(self.input_emb.weight.device.index))

    @torch.jit.ignore
    def save(self, path):
        torch.save(self.state_dict(), path)

    @torch.jit.ignore
    def load(self, path):
        # from ipdb import set_trace
        # set_trace()
        # tmp = torch.load(path, map_location='cpu')
        self.load_state_dict(torch.load(path, map_location='cpu'))


class UpsampleNet(nn.Module):
    def __init__(self, upsample_scales=[2, 2, 4], in_channels=80, out_channels=80, kernel_size=3):
        super(UpsampleNet, self).__init__()
        self._conv = nn.ModuleList()
        ic = in_channels
        for ii in range(3):
            conv = nn.Conv1d(ic, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            ic = out_channels
            self._conv.append(conv)
            self._conv.append(nn.Tanh())
        self._upsample_conv = nn.ModuleList()
        ic = out_channels
        for s in upsample_scales:
            convt = nn.ConvTranspose1d(ic, out_channels, 2 * s, padding=s // 2, stride=(s))
            ic = out_channels
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self._upsample_conv.append(convt)
            self._upsample_conv.append(nn.Tanh())

    def forward(self, c):
        # B x C x T'
        for f in self._conv:
            c = f(c)
        for f in self._upsample_conv:
            c = f(c)
        return c


class UpsampleNetI(nn.Module):
    def __init__(self, upsample=10):
        super(UpsampleNetI, self).__init__()
        self._upsample = upsample

    def forward(self, c):
        # B x C x T'
        ups = torch.nn.functional.interpolate(c, self._upsample * c.shape[2], mode='linear')
        return ups


class UpsampleNet2(nn.Module):
    def __init__(self, upsample_scales=[2, 2, 2, 2], in_channels=80, out_channels=80):
        super(UpsampleNet2, self).__init__()
        self._upsample_conv = nn.ModuleList()
        for s in upsample_scales:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self._upsample_conv.append(convt)
            self._upsample_conv.append(nn.LeakyReLU(0.4))

    def forward(self, c):
        # B x 1 x C x T'
        c = c.unsqueeze(1)
        for f in self._upsample_conv:
            c = f(c)
        # B x C x T
        c = c.squeeze(1)
        return c


class UpsampleNetR(nn.Module):
    def __init__(self, upsample):
        super(UpsampleNetR, self).__init__()
        self._usc = upsample

    def forward(self, c):
        c = c.permute(0, 2, 1)
        c = c.unsqueeze(2)
        c = c.repeat(1, 1, self._usc, 1)
        c = c.reshape(c.shape[0], -1, c.shape[3])
        c = c.permute(0, 2, 1)
        return c


class WaveRNN(nn.Module):
    def __init__(self,
                 num_layers: int = 2,
                 layer_size: int = 512,
                 upsample=100,
                 upsample_low=10,
                 use_lowres=True,
                 learning_rate=1e-4,
                 output='mol'):
        super(WaveRNN, self).__init__()
        # hardcode upsample layers
        # if upsample == 240:
        #     self._upsample_lowres_i = UpsampleNetI(upsample_low)
        #     upsample = [5, 4, 4, 3]  # 240
        #     upsample_low = [2, 5]
        # else:
        #     upsample = [2, 2, 2, 3]  # 24
        self._learning_rate = learning_rate
        # self._upsample_mel = UpsampleNet(upsample_scales=upsample, in_channels=80, out_channels=80)
        self._upsample_mel = UpsampleNetR(upsample=upsample)
        self._use_lowres = use_lowres
        if self._use_lowres:
            self._upsample_lowres = UpsampleNetR(upsample=upsample_low)
            self._lowres_conv = nn.ModuleList()
            ic = 1
            for ii in range(3):
                self._lowres_conv.append(ConvNorm(ic, 20, kernel_size=7, padding=3))
                ic = 20
        ic = 80 + 1
        if use_lowres:
            ic += 21
        self._skip = LinearNorm(ic, layer_size, w_init_gain='tanh')
        rnn_list = []
        for ii in range(num_layers):
            rnn = nn.GRU(input_size=ic, hidden_size=layer_size, num_layers=1, batch_first=True)
            ic = layer_size
            rnn_list.append(rnn)
        self._rnns = nn.ModuleList(rnn_list)
        self._preoutput = LinearNorm(layer_size, 256)

        if output == 'mol':
            self._output_functions = MOLOutput()
        elif output == 'gm':
            self._output_functions = GaussianOutput()
        elif output == 'beta':
            self._output_functions = BetaOutput()
        elif output == 'mulaw':
            self._output_functions = MULAWOutput()
        elif output == 'raw':
            self._output_functions = RAWOutput()

        self._output = LinearNorm(256, self._output_functions.sample_size, w_init_gain='linear')
        self._val_loss = 9999

    def forward(self, X):
        if 'x' in X:
            return self._train_forward(X)
        else:
            return self._inference(X)

    def _inference(self, X):
        with torch.no_grad():
            mel = X['mel']
            if self._use_lowres:
                low_x = X['x_low']
                interp_x = self._upsample_lowres_i(low_x.unsqueeze(1)).permute(0, 2, 1)
                hidden = low_x.unsqueeze(1)
                for conv in self._lowres_conv:
                    hidden = torch.tanh(conv(hidden))
                upsampled_x = self._upsample_lowres(hidden).permute(0, 2, 1)

            upsampled_mel = self._upsample_mel(mel.permute(0, 2, 1)).permute(0, 2, 1)
            cond = upsampled_mel
            if self._use_lowres:
                msize = min(upsampled_mel.shape[1], upsampled_x.shape[1], interp_x.shape[1])
                cond = torch.cat([upsampled_mel[:, :msize, :],
                                  upsampled_x[:, :msize, :],
                                  interp_x[:, :msize, :]],
                                 dim=-1)

            last_x = torch.ones((cond.shape[0], 1, 1),
                                device=self._get_device()) * 0  # * self._x_zero
            output_list = []
            hxs = [None for _ in range(len(self._rnns))]
            # index = 0
            for ii in tqdm.tqdm(range(cond.shape[1]), ncols=80):
                hidden = cond[:, ii, :].unsqueeze(1)
                # res = self._skip(torch.cat([cond[:, ii, :].unsqueeze(1), last_x], dim=-1))
                hidden = torch.cat([hidden, last_x], dim=-1)
                for ll in range(len(self._rnns)):
                    rnn_input = hidden  # torch.cat([hidden, last_x], dim=-1)
                    rnn = self._rnns[ll]
                    rnn_output, hxs[ll] = rnn(rnn_input, hx=hxs[ll])
                    hidden = rnn_output
                    # res = res + hidden

                preoutput = torch.tanh(self._preoutput(hidden))
                output = self._output(preoutput)
                output = output.reshape(output.shape[0], -1, self._output_functions.sample_size)
                samples = self._output_functions.sample(output)
                # if self._use_lowres and ii < interp_x.shape[1]:
                #     last_x = (samples + interp_x[:, ii, :]).unsqueeze(1)
                # else:
                last_x = samples.unsqueeze(1)
                output_list.append(samples.unsqueeze(1))

        output_list = torch.cat(output_list, dim=1)
        # if self._use_lowres:
        #     min_s = min(interp_x.shape[1], output_list.shape[1])
        #     output_list = output_list[:, :min_s].squeeze() + interp_x[:, :min_s].squeeze()
        return output_list.detach().cpu().numpy()  # self._output_functions.decode(output_list)

    def _train_forward(self, X):
        mel = X['mel']
        gs_x = X['x']

        upsampled_mel = self._upsample_mel(mel.permute(0, 2, 1)).permute(0, 2, 1)
        # check if we are using lowres signal conditioning
        if self._use_lowres:
            low_x = X['x_low']
            interp_x = self._upsample_lowres_i(low_x.unsqueeze(1)).squeeze(1)
            hidden = low_x.unsqueeze(1)
            for conv in self._lowres_conv:
                hidden = torch.tanh(conv(hidden))

            upsampled_x = self._upsample_lowres(hidden).permute(0, 2, 1)
            msize = min(upsampled_mel.shape[1], gs_x.shape[1], upsampled_x.shape[1], interp_x.shape[1])
            upsampled_x = upsampled_x[:, :msize, :]
        else:
            msize = min(upsampled_mel.shape[1], gs_x.shape[1])

        upsampled_mel = upsampled_mel[:, :msize, :]
        gs_x = gs_x[:, :msize].unsqueeze(2)
        if self._use_lowres:
            hidden = torch.cat([upsampled_mel, upsampled_x, interp_x.unsqueeze(2), gs_x], dim=-1)
        else:
            hidden = torch.cat([upsampled_mel, gs_x], dim=-1)
        # res = self._skip(hidden)

        for ll in range(len(self._rnns)):
            rnn_input = hidden
            rnn_output, _ = self._rnns[ll](rnn_input)
            hidden = rnn_output
            # res = res + hidden
        preoutput = torch.tanh(self._preoutput(hidden))
        output = self._output(preoutput)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        gs_audio = batch['x']
        target_x = gs_audio[:, 1:]
        pred_x = output[:, :-1]
        loss = self._output_functions.loss(pred_x, target_x)
        return loss

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        gs_audio = batch['x']
        target_x = gs_audio[:, 1:]
        pred_x = output[:, :-1]
        loss = self._output_functions.loss(pred_x, target_x)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        loss = sum(outputs) / len(outputs)
        self.log("val_loss", loss)
        self._val_loss = loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._learning_rate)

    @torch.jit.ignore
    def _get_device(self):
        if self._output.linear_layer.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._output.linear_layer.weight.device.type,
                                str(self._output.linear_layer.weight.device.index))

    @torch.jit.ignore
    def save(self, path):
        torch.save(self.state_dict(), path)

    @torch.jit.ignore
    def load(self, path):
        # from ipdb import set_trace
        # set_trace()
        # tmp = torch.load(path, map_location='cpu')
        self.load_state_dict(torch.load(path, map_location='cpu'))
