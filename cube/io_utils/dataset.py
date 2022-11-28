import scipy
import scipy.io.wavfile
import numpy as np
import sys


class Encodings:
    def __init__(self):
        self.char2int = {}
        self.context2int = {}
        self.speaker2int = {}

    def update(self, pi):
        if pi.char not in self.char2int:
            self.char2int[pi.char] = len(self.char2int)
        for feature in pi.context:
            if not feature.startswith("SPEAKER:"):
                if feature not in self.context2int:
                    self.context2int[feature] = len(self.context2int)
            else:
                if feature not in self.speaker2int:
                    self.speaker2int[feature] = len(self.speaker2int)

    def store(self, filename):
        f = open(filename, 'w')
        f.write('SYMBOLS\t' + str(len(self.char2int)) + '\n')
        for char in self.char2int:
            f.write(char + '\t' + str(self.char2int[char]) + '\n')
        f.write('FEATURES\t' + str(len(self.context2int)) + '\n')
        for feature in self.context2int:
            f.write(feature + '\t' + str(self.context2int[feature]) + '\n')
        f.write('SPEAKERS\t' + str(len(self.speaker2int)) + '\n')
        for feature in self.speaker2int:
            f.write(feature + '\t' + str(self.speaker2int[feature]) + '\n')
        f.close()

    def load(self, filename):
        f = open(filename, encoding='utf-8')
        num_symbols = int(f.readline().split('\t')[1])
        for x in range(num_symbols):
            parts = f.readline().split('\t')
            self.char2int[parts[0]] = int(parts[1])

        num_features = int(f.readline().split('\t')[1])
        for x in range(num_features):
            parts = f.readline().split('\t')
            self.context2int[parts[0]] = int(parts[1])

        num_speakers = int(f.readline().split('\t')[1])
        for x in range(num_speakers):
            parts = f.readline().split('\t')
            self.speaker2int[parts[0]] = int(parts[1])
        f.close()


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

    def write_wave(self, filename, data, sample_rate, dtype=np.float):
        wav_decoded = np.asarray(data, dtype=dtype)
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
            lines = f.readlines()
            for line in lines:
                line = line.replace('\r', '').replace('\n', '')
                if line.strip() != '':
                    line = line
                    parts = line.split('\t')
                    if len(parts) == 1:
                        pi = PhoneInfo(parts[0], [], 0, 0)
                    else:
                        pi = PhoneInfo(parts[0], parts[1:], 0, 0)
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

    def b16_to_float(self, data, discreete=True):
        out = []
        for zz in range(len(data)):
            out.append(float(data[zz]) / 32768)
        return out

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
            encoded = np.clip(encoded, -1.0, 1.0)
            encoded_d = np.clip(encoded_d, 0, 255)
            out_discreete.append(int(encoded_d))
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
            if file[-4:] == '.txt' and base_name not in final_list:
                final_list.append(join(folder, base_name))
                # sys.stdout.write(base_name + '\n')
        self.files = final_list


class PhoneInfo:
    context2int = {}

    def __init__(self, char, context, start, stop):
        self.char = char
        self.context = context
        self.start = start
        self.stop = stop
        self.duration = (stop - start)


class LTSDataset:
    def __init__(self, filename):
        f = open(filename)
        lines = f.readlines()
        self.entries = []
        for line in lines:
            line = ''.join([i for i in line if not i.isdigit()]).strip()
            parts = line.replace('\t', ' ').split(' ')
            word = parts[0].lower()
            transcription = parts[1:]
            self.entries.append(LSTEntry(word, transcription))
        f.close()


class LSTEntry:
    def __init__(self, word, transcription):
        self.word = word
        self.transcription = transcription
