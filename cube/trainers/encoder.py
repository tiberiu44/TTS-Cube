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
    def __init__(self, vocoder, trainset, devset):
        self.vocoder = vocoder
        self.trainset = trainset
        self.devset = devset
        self.mean = None
        self.stdev = None

    def array2file(self, a, filename):
        np.save(filename, a)

    def synth_devset(self, max_size=-1):
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
            lab_file = file + ".txt"
            dio = DatasetIO()
            lab = dio.read_lab(lab_file)
            phones = [entry.phoneme for entry in lab]
            import time
            start = time.time()
            mgc, att = self.vocoder.generate(phones, max_size=max_size)
            mgc = self._denormalize(mgc, mean=self.mean, stdev=self.stdev)

            self.array2file(self._denormalize(mgc, mean=self.mean, stdev=self.stdev),
                            'data/output/' + file[file.rfind('/') + 1:] + '.mgc')
            mgc = self._normalize(mgc, mean=self.mean, stdev=self.stdev)
            att = [a.value() for a in att]
            new_att = np.zeros((len(att), len(phones) + 2, 3), dtype=np.uint8)

            for ii in range(len(phones) + 2):
                for jj in range(len(att)):
                    val = np.clip(int(att[jj][ii] * 255), 0, 255)
                    new_att[jj, ii, 0] = val
                    new_att[jj, ii, 1] = val
                    new_att[jj, ii, 2] = val

            from PIL import Image
            img = Image.fromarray(new_att, 'RGB')
            img.save('data/output/' + file[file.rfind('/') + 1:] + 'att.png')

            output_file = 'data/output/' + file[file.rfind('/') + 1:] + '.png'
            bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
            for x in xrange(mgc.shape[0]):
                for y in xrange(mgc.shape[1]):
                    val = mgc[x, y]
                    color = np.clip(val * 255, 0, 255)
                    bitmap[y, x] = [color, color, color]
            import scipy.misc as smp
            img = smp.toimage(bitmap)
            img.save(output_file)
            stop = time.time()
            sys.stdout.write(" execution time=" + str(stop - start))
            sys.stdout.write('\n')
            sys.stdout.flush()

    def _normalize(self, mgc, mean, stdev):
        for x in xrange(mgc.shape[0]):
            mgc[x] = np.clip((mgc[x] - self.min_db) / (self.max_db - self.min_db), 1e-8, 1.0)
        return mgc

    def _denormalize(self, mgc, mean, stdev):
        for x in xrange(mgc.shape[0]):
            mgc[x] = mgc[x] * (self.max_db - self.min_db) + self.min_db
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

    def start_training(self, itt_no_improve, batch_size, params):
        epoch = 1
        left_itt = itt_no_improve
        dio = DatasetIO()

        sys.stdout.write("Computing mean and standard deviation for spectral parameters\n")
        file_index = 1
        mean = None
        stdev = None
        count = 0
        min_db = None
        max_db = None
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
                max_val = frame[np.argmax(frame)]
                min_val = frame[np.argmin(frame)]

                if min_db is None or min_val < min_db:
                    min_db = min_val
                if max_db is None or max_val > max_db:
                    max_db = max_val
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
        self.min_db = min_db
        self.max_db = max_db
        self._render_devset()
        sys.stdout.write("\n")
        print 'mean =', mean
        print 'stdev =', stdev
        print 'min_db =', min_db
        print 'max_db =', max_db
        if params.no_bounds:
            max_mgc = 1000
        else:
            max_mgc = -1
        self.synth_devset(max_size=max_mgc)
        np.save('data/models/mean_encoder', self.mean)
        np.save('data/models/stdev_encoder', self.stdev)
        with open('data/models/min_max_encoder', 'w') as f:
            f.write(str(min_db) + ' ' + str(max_db) + '\n')
            f.close()
        self.vocoder.store('data/models/rnn_encoder')
        # self.synth_devset(batch_size, target_sample_rate)
        while left_itt > 0:
            sys.stdout.write("Starting epoch " + str(epoch) + "\n")
            sys.stdout.write("Shuffling training data\n")
            from random import shuffle
            shuffle(self.trainset.files)
            file_index = 1
            total_loss = 0
            for file in self.trainset.files:
                sys.stdout.write(
                    "\t" + str(file_index) + "/" + str(len(self.trainset.files)) + " processing file " + file)
                sys.stdout.flush()

                mgc_file = file + ".mgc.npy"
                mgc = np.load(mgc_file)

                lab_file = file + ".txt"
                lab = dio.read_lab(lab_file)
                phones = [entry.phoneme for entry in lab]
                # custom normalization - we are now using binary divergence
                mgc = self._normalize(mgc, mean, stdev)
                file_index += 1

                import time
                start = time.time()
                if len(mgc) < 2000:
                    loss = self.vocoder.learn(phones, mgc, guided_att=not params.no_guided_attention)
                else:
                    sys.stdout.write(' too long, skipping')
                    loss = 0
                total_loss += loss
                stop = time.time()
                sys.stdout.write(' avg loss=' + str(loss) + " execution time=" + str(stop - start))
                sys.stdout.write('\n')
                sys.stdout.flush()
                if file_index % 200 == 0:
                    self.synth_devset(batch_size)
                    self.vocoder.store('data/models/rnn_encoder')

            self.synth_devset(batch_size)
            self.vocoder.store('data/models/rnn_encoder')

            epoch += 1
