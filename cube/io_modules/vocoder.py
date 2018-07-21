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

import pyworld as pw
import numpy as np
import librosa
from scipy import signal


class WorldVocoder:

    def extract_spectrum(self, x, sample_rate):
        x = np.asarray(x)
        _f0, t = pw.dio(x, sample_rate, frame_period=12.5)  # raw pitch extractor
        f0 = pw.stonemask(x, _f0, t, sample_rate)  # pitch refinement
        sp = pw.cheaptrick(x, f0, t, sample_rate)  # extract smoothed spectrogram
        ap = pw.d4c(x, f0, t, sample_rate)
        return sp, ap, f0

    def synthesize(self, sp, ap, f0, sample_rate):
        y = pw.synthesize(f0, sp, ap, sample_rate, frame_period=12.5)
        return y


class MelVocoder:
    def __init__(self):
        self._mel_basis = None

    def melspectrogram(self, y, sample_rate, num_mels):
        pre_y = self.preemphasis(y)
        D = self._stft(pre_y, sample_rate)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D), sample_rate, num_mels))
        return self._normalize(S).transpose()

    def preemphasis(self, x):
        return signal.lfilter([1, -0.97], [1], x)

    def _stft(self, y, sample_rate):
        n_fft, hop_length, win_length = self._stft_parameters(sample_rate)
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def _linear_to_mel(self, spectrogram, sample_rate, num_mels):
        if self._mel_basis is None:
            self._mel_basis = self._build_mel_basis(sample_rate, num_mels)
        return np.dot(self._mel_basis, spectrogram)

    def _build_mel_basis(self, sample_rate, num_mels):
        n_fft = (513 - 1) * 2
        return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels)

    def _normalize(self, S):
        min_level_db = -100.0
        return np.clip((S - min_level_db) / -min_level_db, 0, 1)

    def _stft_parameters(self, sample_rate):
        n_fft = (513 - 1) * 2
        hop_length = int(12.5 / 1000 * sample_rate)
        win_length = int(50.0 / 1000 * sample_rate)
        return n_fft, hop_length, win_length

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))
