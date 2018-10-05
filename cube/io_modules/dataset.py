#
# Author: Tiberiu Boros
# Edited: Adriana Stan, october 2018
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

import scipy
import scipy.io.wavfile
import numpy as np

class DatasetIO:

    def __init__(self, features):
        self._mel_basis = None
        self.features = features

    def read_input_feats(self, filename, feature_set):
        input_feats = []
        with open(filename) as f:
            for line in f.readlines():
                tmp = []
                data = line.strip().split()
                index = 0
                for feat in feature_set.get_input_features():
                    if feat.get_category() == 'D':
                        feat.update_discrete2int(data[index])
                        tmp.append(feat.discrete2int[data[index]])
                    if feat.get_category()in ['B', 'R']:
                        tmp.append(data[index])
                    if feat.get_category()  == 'A':
                        tmp.extend([float(x) for x in data[index:index+feat.get_size()]])
                    index += feat.get_size()
                input_feats.append(tmp)
        return input_feats

