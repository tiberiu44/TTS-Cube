import scipy
import scipy.io.wavfile
import numpy as np
import sys


class DatasetIO:
    def __init__(self):
        self._mel_basis = None

    def read_wave(self, filename, sample_rate=None):
        if sample_rate is None:
            sr, wav = scipy.io.wavfile.read(filename)
            wav = np.asarray(wav, dtype=np.float)
            if wav.dtype != np.float and wav.dtype != np.double:
                wav = wav / 32768
        else:
            import librosa
            wav, sr = librosa.load(filename, sr=sample_rate)
            wav = np.asarray(wav, dtype=np.float)
        return wav, sr

    def write_wave(self, filename, data, sample_rate):
        wav_decoded = np.asarray(data, dtype=np.float)
        scipy.io.wavfile.write(filename, sample_rate, wav_decoded)

    def read_phs(self, filename):
        out = []
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "")
                parts = line.split(" ")
                start = int(parts[0]) / 10000
                stop = int(parts[1]) / 10000
                pp = parts[2].split(":")
                phon = pp[0]
                context = parts[2][parts[2].find(":") + 2:]
                phon = phon.split("-")[1]
                phon = phon.split("+")[0]
                pi = PhoneInfo(phon, context, start, stop)
                out.append(pi)
        return out

    def read_lab(self, filename):
        out = []
        with open(filename) as f:
            line = f.readline()
            line = line.decode('utf-8').replace('\n', '').replace('\r', '').replace('\t', ' ').lower()
            for char in line:
                pi = PhoneInfo(char, '', 0, 0)
                out.append(pi)
            f.close()
        return out

    def b16_enc(self, data):
        out_disc = []
        for zz in range(len(data)):
            disc = int((data[zz] + 1.0) * 32767)
            if disc > 65535:
                disc = 65535
            elif disc < 0:
                disc = 0
            out_disc.append(disc)
        return out_disc

    def b16_dec(self, data, discreete=True):
        out = []
        for zz in range(len(data)):
            out.append(float(data[zz]) / 32768 - 1.0)
        return out

    def ulaw_encode(self, data):
        out_discreete = []
        out_continous = []
        for zz in range(len(data)):
            f = float(data[zz])
            sign = np.sign(f)
            encoded = sign * np.log(1.0 + 255.0 * np.abs(f)) / np.log(1.0 + 255.0)
            encoded_d = int((encoded + 1) * 127)
            out_discreete.append(encoded_d)
            out_continous.append(encoded)

        return [out_discreete, out_continous]

    def ulaw_decode(self, data, discreete=True):
        out = []
        for zz in range(len(data)):
            if discreete:
                f = float(data[zz]) / 128 - 1.0
            else:
                f = data[zz]
            sign = np.sign(f)
            decoded = sign * (1.0 / 255.0) * (pow(1.0 + 255, abs(f)) - 1.0)
            # decoded = int(decoded * 32768)
            out.append(decoded)
        return out


class Dataset:
    def __init__(self, folder):
        from os import listdir
        from os.path import isfile, join
        from os.path import exists
        train_files_tmp = [f for f in listdir(folder) if isfile(join(folder, f))]

        final_list = []
        for file in train_files_tmp:
            base_name = file[:-4]
            if file[-4:] == '.lab' and base_name not in final_list:
                final_list.append(join(folder, base_name))
                # sys.stdout.write(base_name + '\n')
        self.files = final_list


class PhoneInfo:
    context2int = {}

    def __init__(self, phoneme, context, start, stop):
        self.phoneme = phoneme
        self.context = context.split("/")
        self.start = start
        self.stop = stop
        self.duration = (stop - start)
