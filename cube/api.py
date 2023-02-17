import os
import sys
import yaml
from pathlib import Path

sys.path.append('')
from cube.io_utils.repository import download_model
from cube.networks.cubegan import Cubegan
from cube.io_utils.io_cubegan import CubeganEncodings, CubeganCollate
from cube.io_utils.io_text import Text2FeatBlizzard


class TTSCube:
    def __init__(self, model_path: str):
        encodings = CubeganEncodings('{0}.encodings'.format(model_path))
        conf = yaml.load(open('{0}.yaml'.format(model_path)), yaml.Loader)
        cond_type = conf['conditioning']
        self._model = Cubegan(encodings, conditioning=cond_type, train=True)
        self._collate = CubeganCollate(encodings, conditioning_type=cond_type)
        self._text2feat = Text2FeatBlizzard()

    @staticmethod
    def load(model_name: str):
        base_name = '{0}/.ttscube/models/{1}'.format(str(Path.home()), model_name)
        if not os.path.exists(base_name):
            os.makedirs(base_name, exist_ok=True)
            download_model(base_name, model_name)
        return TTSCube('{0}/model'.format(base_name))


if __name__ == '__main__':
    model = TTSCube.load('blizzard2023')
    audio = model('')
