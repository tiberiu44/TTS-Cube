import torch
import torch.nn as nn
import pytorch_lightning as pl

import sys
import yaml

sys.path.append('')

from cube.networks.modules import LinearNorm, ConvNorm


class CubenetVocoder(pl.LightningModule):
    def __init__(self, num_layers: int = 2, layer_size: int = 512, psamples: int = 16, stride: int = 16, upsample=256):
        super(CubenetVocoder, self).__init__()

        self._config = {
            'num_layers': num_layers,
            'layer_size': layer_size,
            'psamples': psamples,
            'stride': stride,
            'upsample': upsample
        }
        self.w = LinearNorm(10, 10)

    def forward(self, X):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs) -> None:
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

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
