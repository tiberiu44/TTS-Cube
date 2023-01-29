import torch
import torch.nn as nn
import pytorch_lightning as pl
import sys

sys.path.append('')
from cube.io_utils.io_text2mel import Text2MelEncodings
from cube.networks.modules import ConvNorm, LinearNorm


class CubenetText2Mel(pl.LightningModule):
    def __init__(self, encodings: Text2MelEncodings):
        super(CubenetText2Mel, self).__init__()
        PHON_EMB_SIZE = 64
        SPEAKER_EMB_SIZE = 128
        CHAR_CNN_SIZE = 256
        CHAR_CNN_KS = 3
        CHAR_CNN_NL = 3
        CHAR_RNN_NL = 2
        CHAR_RNN_SIZE = 256
        OVERLAY_RNN_LAYERS = 2
        OVERLAY_RNN_SIZE = 512
        self._encodings = encodings
        # phoneme embeddings
        self._phon_emb = nn.Embedding(len(encodings.phon2int), PHON_EMB_SIZE)
        # speaker embeddings
        self._speaker_emb = nn.Embedding(len(encodings.speaker2int), SPEAKER_EMB_SIZE)
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
        # rnn over the upsampled data
        self._rnn_overlay = nn.LSTM(input_size=CHAR_RNN_SIZE * 2, hidden_size=OVERLAY_RNN_SIZE, )

    def forward(self, X):
        pass

    def training_step(self, batch, batch_ids):
        pass

    def validation_step(self, batch, batch_ids):
        pass

    def validation_epoch_end(self, outputs: []) -> None:
        pass

    def configure_optimizers(self):
        return None

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
        if self._output.linear_layer.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._output.linear_layer.weight.device.type,
                                str(self._output.linear_layer.weight.device.index))
