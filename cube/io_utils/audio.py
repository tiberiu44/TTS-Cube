import random

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import librosa


# _fx = (
#     AudioEffectsChain()
#     .reverb()
# )
#
# _fx2 = (
#     AudioEffectsChain()
#     .highshelf()
#     .reverb()
#     .phaser()
#     .lowshelf()
# )


def _add_reverb(x_raw, sample_rate):
    if random.random() < 0.5:
        rez = torchaudio.sox_effects.apply_effects_tensor(x_raw,
                                                          sample_rate,
                                                          [['reverb'],
                                                           ['phaser']])[0][0, :]
        # return _fx2(x_raw)
    else:
        rez = torchaudio.sox_effects.apply_effects_tensor(x_raw,
                                                          sample_rate,
                                                          [['reverb']])[0][0, :]

    return rez  # rez[0, :] + rez[1, :]


def _add_noise(x_raw, level=0.01):
    if random.random() < 0.5:
        noise = torch.randn_like(x_raw) * level  # np.random.normal(0, level, x_raw.shape[0])
    else:
        noise = torch.rand_like(x_raw) * 2 * level - level  # 0 - level, 0 + level, x_raw.shape[0])
    return x_raw + noise


_noise_files = [join('data/noise', f) for f in listdir('data/noise/') if
                isfile(join('data/noise', f)) and f.endswith('.wav')]


def _add_real_noise(x_raw, orig_sr=48000):
    while True:
        noise_file = _noise_files[random.randint(0, len(_noise_files) - 1)]
        file_size = os.path.getsize(noise_file)
        if file_size > orig_sr:
            break
    noise_audio, c_sr = torchaudio.load(noise_file)  # librosa.load(noise_file, sr=orig_sr, mono=True)
    noise_audio = noise_audio[0, :].unsqueeze(0)
    resampler = T.Resample(c_sr, orig_sr, dtype=noise_audio.dtype)
    noise_audio = resampler(noise_audio)
    noise_audio = (noise_audio / (torch.max(torch.abs(noise_audio)))) * (random.random() / 4 + 0.2)
    while noise_audio.shape[1] < x_raw.shape[1]:
        noise_audio = torch.cat([noise_audio, noise_audio], dim=1)
    noise_audio = noise_audio[:, :x_raw.shape[1]]
    return x_raw + noise_audio


def _downsample(x_raw, orig_sr):
    p = random.random()
    if p < 0.5:
        sr = 8000
    elif p < 0.7:
        sr = 16000
    elif p < 0.9:
        sr = 22050
    else:
        sr = 24000
    resampler1 = T.Resample(orig_sr, sr, dtype=x_raw.dtype)
    resampler2 = T.Resample(sr, orig_sr, dtype=x_raw.dtype)
    x = resampler1(x_raw)
    x = resampler2(x)
    # x = librosa.resample(x_raw, orig_sr=orig_sr, target_sr=sr)
    # x = librosa.resample(x, orig_sr=sr, target_sr=orig_sr)
    return x


def alter(x_raw, prob=0.1, real_sr=48000):
    # p = random.random()
    # if p < prob:
    #    x_raw = _add_real_noise(x_raw, orig_sr=real_sr)
    p = random.random()
    if p < prob:
        x_raw = _add_reverb(x_raw, real_sr)
    p = random.random()
    if p < prob:
        x_raw = _add_noise(x_raw)

    p = random.random()
    if p < prob:
        x_raw = _downsample(x_raw, orig_sr=real_sr)

    return x_raw
