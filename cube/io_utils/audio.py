import random

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import librosa
import numpy as np
from torchaudio.transforms import Resample
from torchaudio.functional import highpass_biquad


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


def _phone_simulation(x, sr=48000):
    x, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        x,
        sr,
        effects=[
            ["lowpass", "4000"],
            ["compand", "0.02,0.05", "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8", "-8", "-7", "0.05"],
            ["rate", "8000"],
        ],
    )
    x = torchaudio.functional.apply_codec(x, sample_rate, format='gsm')
    resampler = T.Resample(8000, sr, dtype=x.dtype)
    x = resampler(x)

    return x


def pitch_shift(x, sr=48000):
    # we pitch shift using linear interpolation - it is faster than actual pitch shifting
    sf = (random.random() * 2 - 1.0) * 0.3 + 1.0
    x_out = torch.nn.functional.interpolate(x.unsqueeze(1), scale_factor=sf, mode='linear')
    return x_out.squeeze(1)


# code taken from https://github.com/tencent-ailab/FRA-RIR
def fra_rir(nsource=1, sr=16000, direct_range=[-6, 50], max_T60=0.8,
            alpha=0.25, a=-2.0, b=2.0, tau=0.2):
    """
    The fast random approximation of room impulse response (FRA-RIR) method.
    args:
        nsource: number of sources (RIR filters) to simulate. Default: 1.
        sr: target sample rate. Default: 16000.
        direct_range: the context range (at milliseconds) at the first peak of the RIR filter to define the direct-path RIR. Default: [-6, 50] ms.
        max_T60: the maximum range of T60 to sample from. Default: 0.8.
        alpha: controlling the probability distribution to sample the distance of the virtual sound sources from. Default: 0.25.
        a, b: controlling the random pertubation added to each virtual sound source. Default: -2, 2.
        tau: controlling the relationship between the distance and the number of reflections of each virtual sound source. Default: 0.25.
    output:
        rir_filter: simulated RIR filter for all sources, shape: (nsource, nsample)
        direct_rir_filter: simulated direct-path RIR filter for all sources, shape: (nsource, nsample)
    """

    eps = np.finfo(np.float16).eps

    # sample distance between the sound sources and the receiver (d_0)
    direct_dist = torch.FloatTensor(nsource).uniform_(0.2, 12)

    # sample T60 of the room
    T60 = torch.FloatTensor(1).uniform_(0.1, max_T60)[0].data

    # sample room-related statistics for calculating the reflection coefficient R
    R = torch.FloatTensor(1).uniform_(0.1, 1.2)[0].data

    # number of virtual sound sources
    image = sr * 2

    # the sample rate at which the original RIR filter is generated
    ratio = 64
    sample_sr = sr * ratio

    # sound velocity
    velocity = 340.

    # indices of direct-path signals based on the sampled d_0
    direct_idx = torch.ceil(direct_dist * sample_sr / velocity).long()

    # length of the RIR filter based on the sampled T60
    rir_length = int(np.ceil(sample_sr * T60))

    # two resampling operations
    resample1 = Resample(sample_sr, sample_sr // int(np.sqrt(ratio)))
    resample2 = Resample(sample_sr // int(np.sqrt(ratio)), sr)

    # calculate the reflection coefficient based on the Eyring's empirical equation
    reflect_coef = (1 - (1 - torch.exp(-0.16 * R / T60)).pow(2)).sqrt()

    # randomly sample the propagation distance for all the virtual sound sources
    dist_range = [torch.linspace(1., velocity * T60 / direct_dist[i], image) for i in range(nsource)]
    # a simple quadratic function
    dist_prob = torch.linspace(alpha, 1., image).pow(2)
    dist_prob = dist_prob / dist_prob.sum()
    dist_select_idx = dist_prob.multinomial(num_samples=image * nsource, replacement=True).view(nsource, image)
    # the distance is sampled as a ratio between d_0 and each virtual sound sources
    dist_ratio = torch.stack([dist_range[i][dist_select_idx[i]] for i in range(nsource)], 0)
    dist = direct_dist.view(-1, 1) * dist_ratio

    # sample the number of reflections (can be nonintegers)
    # calculate the maximum number of reflections
    reflect_max = (torch.log10(velocity * T60) - torch.log10(direct_dist) - 3) / torch.log10(reflect_coef + eps)
    # calculate the number of reflections based on the assumption that
    # virtual sound sources which have longer propagation distances may reflect more frequently
    reflect_ratio = (dist / (velocity * T60)).pow(2) * (reflect_max.view(nsource, -1) - 1) + 1
    # add a random pertubation based on the assumption that
    # virtual sound sources which have similar propagation distances can have different routes and reflection patterns
    reflect_pertub = torch.FloatTensor(nsource, image).uniform_(a, b) * dist_ratio.pow(tau)
    # all virtual sound sources should reflect for at least once
    reflect_ratio = torch.maximum(reflect_ratio + reflect_pertub, torch.ones(1))

    # calculate the rescaled dirac comb as RIR filter
    dist = torch.cat([direct_dist.reshape(-1, 1), dist], 1)
    reflect_ratio = torch.cat([torch.zeros(nsource, 1), reflect_ratio], 1)
    rir = torch.zeros(nsource, rir_length)
    delta_idx = torch.minimum(torch.ceil(dist * sample_sr / velocity), torch.ones(1) * rir_length - 1).long()
    delta_decay = reflect_coef.pow(reflect_ratio) / dist
    delta_idx, sorted_idx = torch.sort(delta_idx)
    delta_decay = torch.stack([delta_decay[i][sorted_idx[i]] for i in range(nsource)], 0)
    delta_idx = delta_idx.data.cpu().numpy()
    # iteratively detect unique indices and add to the filter
    for i in range(nsource):
        remainder_idx = delta_idx[i]
        valid_mask = np.ones(len(remainder_idx))
        while np.sum(valid_mask) > 0:
            valid_remainder_idx, unique_remainder_idx = np.unique(remainder_idx, return_index=True)
            rir[i][valid_remainder_idx] += delta_decay[i][unique_remainder_idx] * valid_mask[unique_remainder_idx]
            valid_mask[unique_remainder_idx] = 0
            remainder_idx[unique_remainder_idx] = 0

    # a binary mask for direct-path RIR
    direct_mask = torch.zeros(nsource, rir_length).float()
    for i in range(nsource):
        direct_mask[i, max(direct_idx[i] + sample_sr * direct_range[0] // 1000, 0):min(
            direct_idx[i] + sample_sr * direct_range[1] // 1000, rir_length)] = 1.
    rir_direct = rir * direct_mask

    # downsample
    all_rir = torch.stack([rir, rir_direct], 1).view(nsource * 2, -1)
    rir_downsample = resample1(all_rir)

    # apply high-pass filter
    rir_hp = highpass_biquad(rir_downsample, sample_sr // int(np.sqrt(ratio)), 80.)

    # downsample again
    rir = resample2(rir_hp).float().view(nsource, 2, -1)

    # RIR filter and direct-path RIR filter at target sample rate
    rir_filter = rir[:, 0]  # nsource, nsample
    direct_rir_filter = rir[:, 1]  # nsource, nsample

    return rir_filter, direct_rir_filter


def _add_rir(x_raw, real_sr=48000):
    if len(x_raw.size()) == 1:
        x_raw = x_raw.unsqueeze(0)
    rir, direct_rir = fra_rir(nsource=1, sr=real_sr)
    augmented = torchaudio.functional.fftconvolve(x_raw, rir)
    return augmented


def alter(x_raw, prob=0.1, real_sr=48000):
    p = random.random()
    if p < prob:
        x_raw = _add_real_noise(x_raw, orig_sr=real_sr)

    p = random.random()
    if p < prob:
        x_raw = _add_reverb(x_raw, real_sr)

    p = random.random()
    if p < prob:
        x_raw = _add_noise(x_raw)

    p = random.random()
    if p < prob:
        x_raw = _add_rir(x_raw, real_sr)

    p = random.random()
    if p < prob:
        x_raw = _downsample(x_raw, orig_sr=real_sr)

    return x_raw


if __name__ == '__main__':
    audio, sr = torchaudio.load('data/processed/dev/clean/FILE_00000001.wav')
    audio2 = _add_rir(audio, sr)
    import soundfile as sf

    sf.write('tmp.wav', audio2.detach().squeeze().numpy(), 24000, 'PCM_16')
