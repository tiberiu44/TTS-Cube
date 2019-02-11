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

# import pyworld as pw
import numpy as np
import librosa
from scipy import signal
from tqdm import tqdm


class WorldVocoder:

    def extract_spectrum(self, x, sample_rate):
        x = np.asarray(x)
        _f0, t = pw.dio(x, sample_rate, frame_period=16)  # raw pitch extractor
        f0 = pw.stonemask(x, _f0, t, sample_rate)  # pitch refinement
        sp = pw.cheaptrick(x, f0, t, sample_rate)  # extract smoothed spectrogram
        ap = pw.d4c(x, f0, t, sample_rate)
        return sp, ap, f0

    def synthesize(self, sp, ap, f0, sample_rate):
        y = pw.synthesize(f0, sp, ap, sample_rate, frame_period=16)
        return y


class MelVocoder:
    def __init__(self):
        self._mel_basis = None

    def fft(self, y, sample_rate, use_preemphasis=True):
        if use_preemphasis:
            pre_y = self.preemphasis(y)
        else:
            pre_y = y
        D = self._stft(pre_y, sample_rate)
        return D.transpose()

    def ifft(self, y, sample_rate):
        y = y.transpose()
        return self._istft(y, sample_rate)

    def melspectrogram(self, y, sample_rate, num_mels):
        pre_y = self.preemphasis(y)
        D = self._stft(pre_y, sample_rate)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D), sample_rate, num_mels))
        return self._normalize(S).transpose()

    def preemphasis(self, x):
        return signal.lfilter([1, -0.97], [1], x)

    def _istft(self, y, sample_rate):
        n_fft, hop_length, win_length = self._stft_parameters(sample_rate)
        return librosa.istft(y, hop_length=hop_length, win_length=win_length)

    def _stft(self, y, sample_rate):
        n_fft, hop_length, win_length = self._stft_parameters(sample_rate)
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann')

    def _linear_to_mel(self, spectrogram, sample_rate, num_mels):
        if self._mel_basis is None:
            self._mel_basis = self._build_mel_basis(sample_rate, num_mels)
        return np.dot(self._mel_basis, spectrogram)

    def _build_mel_basis(self, sample_rate, num_mels):
        n_fft = 1024
        return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels)

    def _normalize(self, S):
        min_level_db = -100.0
        return np.clip((S - min_level_db) / -min_level_db, 0, 1)

    def _stft_parameters(self, sample_rate):
        n_fft = 1024
        hop_length = 256
        win_length = n_fft
        return n_fft, hop_length, win_length

    def _amp_to_db(self, x):
        reference = 0.0
        return 20 * np.log10(np.maximum(1e-5, x)) - reference

    def griffinlim(self, spectrogram, n_iter=100, sample_rate=16000):
        n_fft, hop_length, win_length = self._stft_parameters(sample_rate)
        return self._griffinlim(spectrogram.transpose(), n_iter=n_iter, n_fft=n_fft, hop_length=hop_length)

    def _griffinlim(self, spectrogram, n_iter=100, window='hann', n_fft=2048, hop_length=-1, verbose=False):
        if hop_length == -1:
            hop_length = n_fft // 4

        angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

        t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
        for i in t:
            full = np.abs(spectrogram).astype(np.complex) * angles
            inverse = librosa.istft(full, hop_length=hop_length, window=window)
            rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length, window=window)
            angles = np.exp(1j * np.angle(rebuilt))

            if verbose:
                diff = np.abs(spectrogram) - np.abs(rebuilt)
                t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length=hop_length, window=window)

        return inverse
