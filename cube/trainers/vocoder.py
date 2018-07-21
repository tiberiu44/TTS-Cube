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

import sys
import numpy as np
from io_modules.dataset import DatasetIO


class Trainer:
    def __init__(self, vocoder, trainset, devset, use_ulaw=False):
        self.vocoder = vocoder
        self.trainset = trainset
        self.devset = devset
        self.mean = None
        self.stdev = None
        self.use_ulaw = use_ulaw

    def synth_devset(self, batch_size, target_sample_rate, sample=True, temperature=1.0):
        if self.mean is None:
            self.mean = np.load('data/models/mean.npy')
            self.stdev = np.load('data/models/stdev.npy')
        sys.stdout.write('\tSynthesizing devset\n')
        file_index = 1
        for file in self.devset.files[:5]:
            sys.stdout.write(
                "\t\t" + str(file_index) + "/" + str(len(self.devset.files)) + " processing file " + file)
            sys.stdout.flush()
            file_index += 1
            mgc_file = file + ".mgc.npy"
            mgc = np.load(mgc_file)
            mgc = self._normalize(mgc, self.mean, self.stdev)
            import time
            start = time.time()
            synth = self.vocoder.synthesize(mgc, batch_size, sample=sample, temperature=temperature)
            stop = time.time()
            sys.stdout.write(" execution time=" + str(stop - start))
            sys.stdout.write('\n')
            sys.stdout.flush()

            dio = DatasetIO()
            if self.use_ulaw:
                enc = dio.ulaw_decode(synth, discreete=True)
            else:
                enc = dio.b16_dec(synth, discreete=True)
            output_file = 'data/output/' + file[file.rfind('/') + 1:] + '.wav'
            dio.write_wave(output_file, enc, target_sample_rate)

    def _normalize(self, mgc, mean, stdev):
        for x in xrange(mgc.shape[0]):
            mgc[x] = (mgc[x] - mean) / stdev
        return mgc

    def _render_devset(self):
        sys.stdout.write('\tRendering devset\n')
        file_index = 1
        for file in self.devset.files[:5]:
            sys.stdout.write(
                "\t\t" + str(file_index) + "/" + str(len(self.devset.files)) + " processing file " + file + " \n")
            sys.stdout.flush()
            file_index += 1
            mgc_file = file + ".mgc.npy"
            mgc = np.load(mgc_file)
            mgc = self._normalize(mgc, self.mean, self.stdev)
            print mgc.shape
            output_file = 'data/output/' + file[file.rfind('/') + 1:] + '.png'
            bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
            for x in xrange(mgc.shape[0]):
                for y in xrange(mgc.shape[1]):
                    val = mgc[x, y]
                    val = val + 2 * self.stdev[y]
                    val = val / 4
                    if val < 0:
                        val = 0
                    if val > 1:
                        val = 1
                    color = val * 255
                    bitmap[y, x] = [color, color, color]
            import scipy.misc as smp
            img = smp.toimage(bitmap)
            img.save(output_file)

    def start_training(self, itt_no_improve, batch_size, target_sample_rate):
        epoch = 1
        left_itt = itt_no_improve
        dio = DatasetIO()
        # self.synth_devset(batch_size, target_sample_rate)
        sys.stdout.write("Computing mean and standard deviation for spectral parameters\n")
        file_index = 1
        mean = None
        stdev = None
        count = 0
        for file in self.trainset.files:
            sys.stdout.write("\r\tFile " + str(file_index) + "/" + str(len(self.trainset.files)))
            sys.stdout.flush()
            mgc_file = file + ".mgc.npy"
            mgc = np.load(mgc_file)
            if mean is None:
                mean = np.zeros((mgc.shape[1]))
                stdev = np.zeros((mgc.shape[1]))
            for frame in mgc:
                mean += frame
            count += mgc.shape[0]
            file_index += 1
        mean /= count
        file_index = 1
        for file in self.trainset.files:
            sys.stdout.write("\r\tFile " + str(file_index) + "/" + str(len(self.trainset.files)))
            sys.stdout.flush()
            mgc_file = file + ".mgc.npy"
            mgc = np.load(mgc_file)
            for frame in mgc:
                stdev += np.power((frame - mean), 2)
            file_index += 1

        stdev /= count
        stdev = np.sqrt(stdev)
        self.mean = mean
        self.stdev = stdev
        self._render_devset()
        sys.stdout.write("\n")
        print 'mean =', mean
        print 'stdev =', stdev
        np.save('data/models/mean', self.mean)
        np.save('data/models/stdev', self.stdev)
        self.vocoder.store('data/models/rnn')
        #self.synth_devset(batch_size, target_sample_rate)
        while left_itt > 0:
            sys.stdout.write("Starting epoch " + str(epoch) + "\n")
            sys.stdout.write("Shuffling training data\n")
            from random import shuffle
            shuffle (self.trainset.files)
            file_index = 1
            total_loss = 0
            for file in self.trainset.files:
                sys.stdout.write(
                    "\t" + str(file_index) + "/" + str(len(self.trainset.files)) + " processing file " + file)
                sys.stdout.flush()
                wav_file = file + ".orig.wav"
                mgc_file = file + ".mgc.npy"
                mgc = np.load(mgc_file)
                mgc = self._normalize(mgc, mean, stdev)
                file_index += 1
                data, sample_rate = dio.read_wave(wav_file)
                if self.use_ulaw:
                    [wave_disc, ulaw_cont] = dio.ulaw_encode(data)
                else:
                    wave_disc = dio.b16_enc(data)
                import time
                start = time.time()
                loss = self.vocoder.learn(wave_disc, mgc, batch_size)
                total_loss += loss
                stop = time.time()
                sys.stdout.write(' avg loss=' + str(loss) + " execution time=" + str(stop - start))
                sys.stdout.write('\n')
                sys.stdout.flush()
                if file_index % 50 == 0:
                    self.synth_devset(batch_size, target_sample_rate)
                    self.vocoder.store('data/models/rnn')

            self.synth_devset(batch_size, target_sample_rate)
            self.vocoder.store('data/models/rnn')

            epoch += 1
