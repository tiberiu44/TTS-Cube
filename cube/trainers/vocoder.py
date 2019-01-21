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
    def __init__(self, vocoder, trainset, devset, use_ulaw=False, target_output_path='data/models/nn_vocoder'):
        self.vocoder = vocoder
        self.trainset = trainset
        self.devset = devset
        self.use_ulaw = use_ulaw
        self.target_output_path = target_output_path

    def synth_devset(self, batch_size, target_sample_rate, sample=True, temperature=1.0):
        sys.stdout.write('\tSynthesizing devset\n')
        file_index = 1
        for file in self.devset.files[:5]:
            sys.stdout.write(
                "\t\t" + str(file_index) + "/" + str(len(self.devset.files)) + " processing file " + file + "\n")
            sys.stdout.flush()
            file_index += 1
            mgc_file = file + ".mgc.npy"
            mgc = np.load(mgc_file)
            import time
            start = time.time()
            synth = self.vocoder.synthesize(mgc, batch_size)
            stop = time.time()
            sys.stdout.write(" execution time=" + str(stop - start))
            sys.stdout.write('\n')
            sys.stdout.flush()

            dio = DatasetIO()

            output_file = 'data/output/' + file[file.rfind('/') + 1:] + '.wav'
            dio.write_wave(output_file, synth, target_sample_rate, dtype=np.int16)

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
            # print mgc.shape
            output_file = 'data/output/' + file[file.rfind('/') + 1:] + '.png'
            bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
            for x in range(mgc.shape[0]):
                for y in range(mgc.shape[1]):
                    val = mgc[x, y]
                    color = val * 255
                    bitmap[y, x] = [color, color, color]
            import scipy.misc as smp
            img = smp.toimage(bitmap)
            img.save(output_file)

    def start_training(self, itt_no_improve, batch_size, target_sample_rate, params=None):
        epoch = 1
        left_itt = itt_no_improve
        dio = DatasetIO()
        self._render_devset()
        sys.stdout.write("\n")
        # self.synth_devset(batch_size, target_sample_rate)
        self.vocoder.store(self.target_output_path)

        num_files = 0
        while left_itt > 0:
            sys.stdout.write("Starting epoch " + str(epoch) + "\n")
            sys.stdout.write("Shuffling training data\n")
            from random import shuffle
            shuffle(self.trainset.files)
            file_index = 1
            total_loss = 0
            for file in self.trainset.files:
                num_files += 1
                sys.stdout.write(
                    "\t" + str(file_index) + "/" + str(len(self.trainset.files)) + " processing file " + file + '\n')
                sys.stdout.flush()
                wav_file = file + ".orig.wav"
                mgc_file = file + ".mgc.npy"
                mgc = np.load(mgc_file)
                file_index += 1
                data, sample_rate = dio.read_wave(wav_file)
                # wave_disc = data * 32768
                wave_disc = np.array(data, dtype=np.float32)

                import time
                start = time.time()
                loss = self.vocoder.learn(wave_disc, mgc, batch_size)
                total_loss += loss
                stop = time.time()
                sys.stdout.write(' avg loss=' + str(loss) + " execution time=" + str(stop - start))
                sys.stdout.write('\n')
                sys.stdout.flush()
                if file_index % 5000 == 0:
                    self.vocoder.store(self.target_output_path)
                    self.synth_devset(batch_size, target_sample_rate)

            self.vocoder.store(self.target_output_path)
            self.synth_devset(batch_size, target_sample_rate)

            epoch += 1
