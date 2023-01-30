import torch
import torch.nn as nn
import pytorch_lightning as pl
import sys

sys.path.append('')
from cube.io_utils.io_textcoder import TextcoderEncodings
from cube.networks.modules import ConvNorm, LinearNorm, PreNet, PostNet


class CubenetTextcoder(pl.LightningModule):
    def __init__(self, encodings: TextcoderEncodings, pframes: int = 3):
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
        # rnn over the upsampled data - this helps with co-articulation of sounds
        self._rnn_overlay = nn.LSTM(input_size=CHAR_RNN_SIZE * 2,
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
        self._dur_output = LinearNorm(DUR_RNN_SIZE * 2, encodings.max_duration)
        # pitch
        # this comes after the textcnn+speaker_emb+external_cond(could be a transformer)
        self._pitch_rnn = nn.LSTM(input_size=CHAR_RNN_SIZE * 2 + SPEAKER_EMB_SIZE + EXTERNAL_COND,
                                  hidden_size=PITCH_RNN_SIZE,
                                  num_layers=PITCH_RNN_LAYERS,
                                  bidirectional=True,
                                  batch_first=True)
        self._pitch_output = LinearNorm(PITCH_RNN_SIZE * 2, encodings.max_pitch)
        # mel
        # this comes after the rnn_overlay
        self._mel_rnn = nn.LSTM(input_size=OVERLAY_RNN_SIZE * 2 + PRENET_SIZE + SPEAKER_EMB_SIZE + EXTERNAL_COND,
                                hidden_size=MEL_RNN_SIZE,
                                num_layers=MEL_RNN_LAYERS,
                                bidirectional=False,
                                batch_first=True)
        self._mel_output = LinearNorm(MEL_RNN_SIZE, MEL_SIZE * pframes)
        self._prenet = PreNet(MEL_SIZE, PRENET_SIZE, PRENET_LAYERS)
        self._postnet = PostNet(MEL_SIZE)

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
