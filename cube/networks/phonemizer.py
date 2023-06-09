import torch
import torch.nn as nn
import pytorch_lightning as pl
import sys
import numpy as np

sys.path.append('')
from cube.io_utils.io_phonemizer import PhonemizerEncodings
from cube.networks.modules import Attention


class CubenetPhonemizer(pl.LightningModule):
    def __init__(self, encodings: PhonemizerEncodings, lr=2e-4):
        super(CubenetPhonemizer, self).__init__()
        self._encodings = encodings
        self._lr = lr
        # remove dummy and add initialization code here
        self._char_emb = nn.Embedding(len(encodings.graphemes), 32)
        self._case_emb = nn.Embedding(2, 8)
        convs = []
        input_size = 40
        for ii in range(3):  # _0_ v1 v2 v3 v4 v5 v6 _0_ ->
            convs.append(nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1))
            convs.append(nn.Tanh())
            input_size = 256
        self._convs = nn.ModuleList(convs)
        self._rnn = nn.LSTM(input_size=256, hidden_size=200, num_layers=2, batch_first=True, bidirectional=True)
        self._output_softmax = nn.Linear(200 * 2, len(encodings.phonemes))
        self._loss_function = nn.CrossEntropyLoss(ignore_index=0)
        self._val_sacc = 0
        self._val_pacc = 0

    def forward(self, X):
        x_char = X['x_char']  # size (batch_size x sequence len) [[0 1 1 2 3 4 2 1]]
        x_case = X['x_case']  # size (batch_size x sequence len)
        h_char = self._char_emb(x_char)  # size (batch_size x sequence len x 32)
        h_case = self._case_emb(x_case)  # size (batch_size x sequence len x 8)
        h = torch.cat([h_char, h_case], dim=-1)
        # for images: initial input is (batch_size x X x Y x bpp) -> permute has to be permute(0,3,1,2)
        h = h.permute(0, 2, 1)
        for conv in self._convs:
            h = conv(h)
        h = h.permute(0, 2, 1)  # (batch_size x sequence len x 256)
        output_lstm, _ = self._rnn(h)

        # add forward code here
        return self._output_softmax(output_lstm)

    def training_step(self, batch, batch_idx):
        y_target = batch['y_phon']  # size (batch_size x sequence len)
        y_pred = self.forward(batch)
        y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], -1)
        y_target = y_target.reshape(-1)
        loss = self._loss_function(y_pred, y_target)
        return loss

    def validation_step(self, batch, batch_idx):
        y_target = batch['y_phon']
        del batch['y_phon']  # this is not actually going to be present during runtime
        y_pred = self.forward(batch)
        y_pred_tmp = y_pred
        y_target_tmp = y_target
        y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], -1)
        y_target = y_target.reshape(-1)
        loss = self._loss_function(y_pred, y_target)
        return {'loss': loss.item(), 'target': y_target_tmp.detach().cpu().numpy(),
                'pred': torch.argmax(y_pred_tmp, dim=-1).detach().cpu().numpy()}

    def validation_epoch_end(self, outputs):
        perr = 0
        serr = 0
        loss = sum([output['loss'] for output in outputs])
        self._val_loss = loss / len(outputs)
        total_phones = 0
        total_seqs = 0
        for output in outputs:
            target_seqs = output['target']
            pred_seqs = output['pred']
            for target, pred in zip(target_seqs, pred_seqs):
                total_seqs += 1
                seq_ok = True
                for t, p in zip(target.squeeze(), pred.squeeze()):
                    if t != 0:
                        total_phones += 1
                    if t != p and t != 0 and p != 0:
                        perr += 1
                        seq_ok = False
                if not seq_ok:
                    serr += 1

        self._val_pacc = 1.0 - (perr / total_phones)
        self._val_sacc = 1.0 - (serr / total_seqs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._lr)

    @torch.jit.ignore
    def save(self, path):
        torch.save(self.state_dict(), path)

    @torch.jit.ignore
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


def _prepare_encoder_data(output_encoder, x_words, index_word):
    output_stack = []
    m_len = 0
    for ii in range(index_word.shape[0]):
        start = 0
        m_word = min(len(x_words[ii]) - 1, index_word[ii])
        start = x_words[ii][m_word]['start']
        stop = x_words[ii][m_word]['stop']
        output_stack.append(output_encoder[ii, start:stop, :])
        l = stop - start
        if l > m_len:
            m_len = l

    for ii in range(len(output_stack)):
        output_stack[ii] = torch.nn.functional.pad(output_stack[ii],
                                                   (0, 0, 0, m_len - output_stack[ii].shape[0])).unsqueeze(0)
    return torch.cat(output_stack, dim=0)


class CubenetPhonemizerM2M(pl.LightningModule):
    def __init__(self, encodings: PhonemizerEncodings, lr=2e-4):
        super(CubenetPhonemizerM2M, self).__init__()
        self._encodings = encodings
        self._lr = lr
        # remove dummy and add initialization code here
        self._char_emb = nn.Embedding(len(encodings.graphemes), 32, padding_idx=0)
        self._case_emb = nn.Embedding(2, 8)
        self._phon_emb = nn.Embedding(len(encodings.phonemes), 32, padding_idx=0)
        convs = []
        input_size = 40
        for ii in range(3):
            convs.append(nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1))
            convs.append(nn.Tanh())
            input_size = 256
        self._convs = nn.ModuleList(convs)
        self._rnn_enc = nn.LSTM(input_size=256, hidden_size=200, num_layers=2, batch_first=True, bidirectional=True)
        self._rnn_dec = nn.LSTM(input_size=400 + 32, hidden_size=200, num_layers=2, batch_first=True,
                                bidirectional=False)
        self._att = Attention(400, 200, kernel_size=3)
        self._output_phon = nn.Linear(200, len(encodings.phonemes))
        self._output_next_word = nn.Linear(200, 20)  # max skip of 20 words
        self._loss_function = nn.CrossEntropyLoss(ignore_index=0)
        self._val_loss = 9999
        self._val_sacc = 0
        self._val_pacc = 0
        self._validation_outputs = []

    def forward(self, X):
        x_char = X['x_char']
        x_case = X['x_case']
        h_char = self._char_emb(x_char)  # size (batch_size x sequence len x 32)
        h_case = self._case_emb(x_case)  # size (batch_size x sequence len x 8)
        h = torch.cat([h_char, h_case], dim=-1)
        # for images: initial input is (batch_size x X x Y x bpp) -> permute has to be permute(0,3,1,2)
        h = h.permute(0, 2, 1)
        for conv in self._convs:
            h = conv(h)
        h = h.permute(0, 2, 1)  # (batch_size x sequence len x 256)
        output_encoder, _ = self._rnn_enc(h)
        output_list_phon = []
        output_list_nw = []
        decoder_output, h_decoder = self._rnn_dec(torch.zeros((x_char.shape[0], 1, 400 + 32), device=x_char.device))
        last_phone = torch.zeros((x_char.shape[0], 1), dtype=torch.long, device=x_char.device)
        index_phon = 0
        index_word = np.zeros((x_char.shape[0]), dtype='long')
        while True:
            # attention
            attention_input = _prepare_encoder_data(output_encoder, X['x_words'], index_word)
            # attention_input = output_encoder
            _, weighted = self._att(decoder_output.squeeze(1), attention_input)
            last_phone_emb = self._phon_emb(last_phone)
            decoder_input = torch.cat([last_phone_emb, weighted.unsqueeze(1)], dim=-1)
            decoder_output, h_decoder = self._rnn_dec(decoder_input, hx=h_decoder)
            phon_out = self._output_phon(decoder_output)
            nw_out = self._output_next_word(decoder_output)
            output_list_phon.append(phon_out)
            output_list_nw.append(nw_out)
            if 'y_phon' in X:
                last_phone = X['y_phon'][:, index_phon].unsqueeze(1)
            else:
                last_phone = torch.argmax(phon_out, dim=-1)

            if 'y_phon' in X:
                exit_condition = index_phon == X['y_phon'].shape[1] - 1
                if exit_condition:
                    break
                index_word += torch.clip(X['y_new_word'][:, index_phon] - 1, 0).detach().cpu().numpy()
            else:
                nw = torch.clip(torch.argmax(nw_out) - 1, 0)
                index_word += nw.detach().cpu().numpy()
                reached_end = True
                for ii, iw in zip(range(len(index_word)), index_word):
                    if iw < len(X['x_words'][ii]):
                        reached_end = False

                exit_condition = (index_phon >= X['x_char'].shape[1] * 2) or reached_end
            index_phon += 1
            if exit_condition:
                break

        # add forward code here
        return torch.cat(output_list_phon, dim=1), torch.cat(output_list_nw, dim=1)

    def training_step(self, batch, batch_idx):
        y_phon_target = batch['y_phon']  # size (batch_size x sequence len)
        y_nw_target = batch['y_new_word']
        y_phon_pred, y_nw_pred = self.forward(batch)

        y_phon_pred = y_phon_pred.reshape(y_phon_pred.shape[0] * y_phon_pred.shape[1], -1)
        y_nw_pred = y_nw_pred.reshape(y_nw_pred.shape[0] * y_nw_pred.shape[1], -1)
        y_phon_target = y_phon_target.reshape(-1)
        y_nw_target = y_nw_target.reshape(-1)
        loss = self._loss_function(y_phon_pred, y_phon_target) + self._loss_function(y_nw_pred, y_nw_target)
        return loss

    def validation_step(self, batch, batch_idx):
        y_phon_target = batch['y_phon']
        y_nw_target = batch['y_new_word']
        del batch['y_phon']  # this is not actually going to be present during runtime
        y_phon_pred, y_nw_pred = self.forward(batch)
        # y_phon_pred_tmp = y_phon_pred
        # y_phon_target_tmp = y_phon_target
        # y_nw_target_tmp = y_nw_target
        # y_nw_pred_tmp = y_nw_pred
        # y_phon_pred, y_nw_pred = y_phon_pred.reshape(y_phon_pred.shape[0] * y_phon_pred.shape[1], -1)
        # y_phon_target = y_phon_target.reshape(-1)
        # loss = self._loss_function(y_phon_pred, y_phon_target) + self._loss_function(y_nw_pred, y_nw_target)
        self._validation_outputs.append(
            {
                'target': y_phon_target.detach().cpu().numpy(),
                'target_nw': y_nw_target.detach().cpu().numpy(),
                'pred': torch.argmax(y_phon_pred, dim=-1).detach().cpu().numpy(),
                'pred_nw': torch.argmax(y_nw_pred, dim=-1).detach().cpu().numpy()
            }
        )
        return {
            'target': y_phon_target.detach().cpu().numpy(),
            'target_nw': y_nw_target.detach().cpu().numpy(),
            'pred': torch.argmax(y_phon_pred, dim=-1).detach().cpu().numpy(),
            'pred_nw': torch.argmax(y_nw_pred, dim=-1).detach().cpu().numpy()
        }

    def on_validation_epoch_end(self):
        perr = 0
        serr = 0
        outputs = self._validation_outputs
        self._validation_outputs = []
        total_phones = 0
        total_seqs = 0
        try:
            for output in outputs:
                target_seqs = output['target']
                pred_seqs = output['pred']
                for target, pred in zip(target_seqs, pred_seqs):
                    total_seqs += 1
                    seq_ok = True
                    for t, p in zip(target.squeeze(), pred.squeeze()):
                        if t != 0:
                            total_phones += 1
                        if t != p and t != 0 and p != 0:
                            perr += 1
                            seq_ok = False
                    if not seq_ok:
                        serr += 1
            self._val_pacc = 1.0 - (perr / total_phones)
            self._val_sacc = 1.0 - (serr / total_seqs)
        except:
            self._val_pacc = 0
            self._val_sacc = 0

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._lr)

    @torch.jit.ignore
    def save(self, path):
        torch.save(self.state_dict(), path)

    @torch.jit.ignore
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
