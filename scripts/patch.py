from os import listdir
from os.path import isfile, join


def patch(mypath):
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.wav')]
    import pysptk
    import librosa
    import numpy as np
    import tqdm

    for file in tqdm.tqdm(onlyfiles):
        source = file
        dest = file.replace('.wav', '.pitch')
        wav, sr = librosa.load(file, sr=24000)
        pitch = pysptk.rapt(wav * 32767, 24000, hopsize=240)
        np.save(open(dest, 'wb'), pitch)


patch('data/processed/dev')
patch('data/processed/train')
