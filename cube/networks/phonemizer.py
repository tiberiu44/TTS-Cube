import torch
import torch.nn as nn
import pytorch_lightning as pl
import sys

sys.path.append('')
from cube.io_utils.io_phonemizer import PhonemizerEncodings


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
        self._val_loss = 9999
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
