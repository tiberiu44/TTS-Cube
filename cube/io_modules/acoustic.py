import numpy as np
import scipy

class Acoustic:

    def render_spectrogram(self, mgc, output_file):

        bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
        mgc_min = mgc.min()
        mgc_max = mgc.max()

        for x in range(mgc.shape[0]):
            for y in range(mgc.shape[1]):
                val = (mgc[x, y] - mgc_min) / (mgc_max - mgc_min)

                color = val * 255
                bitmap[mgc.shape[1] - y - 1, x] = [color, color, color]
        import scipy.misc as smp

        img = smp.toimage(bitmap)
        img.save(output_file)


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