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
        self._dummy = nn.Embedding(len(encodings.graphemes), len(encodings.phonemes))
        self._loss_function = nn.CrossEntropyLoss(ignore_index=0)
        self._val_loss = 9999
        self._val_sacc = 0
        self._val_pacc = 0

    def forward(self, X):
        x_char = X['x_char']
        x_case = X['x_case']
        # add forward code here
        return self._dummy(x_char)

    def training_step(self, batch, batch_idx):
        y_target = batch['y_phon']
        y_pred = self.forward(batch)
        y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], -1)
        y_target = y_target.reshape(-1)
        loss = self._loss_function(y_pred, y_target)
        return loss

    def validation_step(self, batch, batch_idx):
        y_target = batch['y_phon']
        del batch['y_phon']  # this is not actually going to be present during runtime
        y_pred = self.forward(batch)
        y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], -1)
        y_target = y_target.reshape(-1)
        loss = self._loss_function(y_pred, y_target)
        return {'loss': loss.item(), 'target': y_target.detach().cpu().numpy(),
                'pred': torch.argmax(y_pred, dim=-1).detach().cpu().numpy()}

    def validation_epoch_end(self, outputs):
        perr = 0
        serr = 0
        loss = sum([output['loss'] for output in outputs])
        self._val_loss = loss / len(outputs)
        total_phones = 0
        for output in outputs:
            target = output['target']
            pred = output['pred']
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
        self._val_sacc = 1.0 - (serr / len(outputs))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._lr)

    @torch.jit.ignore
    def save(self, path):
        torch.save(self.state_dict(), path)

    @torch.jit.ignore
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
