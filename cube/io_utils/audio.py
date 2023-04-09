from pysndfx import AudioEffectsChain

_fx = (
    AudioEffectsChain()
        .reverb()
)

_fx2 = (
    AudioEffectsChain()
        .highshelf()
        .reverb()
        .phaser()
        .lowshelf()
)


def _add_reverb(x_raw):
    if random.random() < 0.5:
        return _fx2(x_raw)
    else:
        return _fx(x_raw)


def _add_noise(x_raw, level=0.01):
    if random.random() < 0.5:
        noise = np.random.normal(0, level, x_raw.shape[0])
    else:
        noise = np.random.uniform(0 - level, 0 + level, x_raw.shape[0])
    return x_raw + noise


_noise_files = [join('data/noise', f) for f in listdir('data/noise/') if
                isfile(join('data/noise', f)) and f.endswith('.wav')]


def _add_real_noise(x_raw):
    while True:
        noise_file = _noise_files[random.randint(0, len(_noise_files) - 1)]
        file_size = os.path.getsize(noise_file)
        if file_size > 22050:
            break
    noise_audio, _ = librosa.load(noise_file, sr=22050, mono=True)
    noise_audio = (noise_audio / (max(abs(noise_audio)))) * (random.random() / 4 + 0.2)
    while len(noise_audio) < len(x_raw):
        noise_audio = np.concatenate((noise_audio, noise_audio))
    noise_audio = noise_audio[:len(x_raw)]
    return x_raw + noise_audio


def _downsample(x_raw):
    p = random.random()
    if p < 0.5:
        sr = 8000
    else:
        sr = 16000
    x = librosa.resample(x_raw, orig_sr=22050, target_sr=sr)
    x = librosa.resample(x, orig_sr=sr, target_sr=22050)
    return x


def alter(x_raw, prob=0.1):
    p = random.random()
    if p < prob:
        x_raw = _add_reverb(x_raw)
    p = random.random()
    if p < prob:
        x_raw = _add_noise(x_raw)
    p = random.random()
    if p < prob:
        x_raw = _add_real_noise(x_raw)

    p = random.random()
    if p < prob:
        x_raw = _downsample(x_raw)

    return x_raw
