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
    def __init__(self, vocoder, features, setup_trainer):
        self.vocoder = vocoder
        self.features = features
        self.setup_trainer = setup_trainer
        self.trainset = self.setup_trainer.train_file_list
        self.devset = self.setup_trainer.dev_file_list

    def array2file(self, a, filename):
        np.save(filename, a)

    def synth_devset(self, max_size=-1):
        sys.stdout.write('\tSynthesizing devset\n')
        file_index = 1
        for file in self.devset[:5]:
            sys.stdout.write(
                "\t\t" + str(file_index) + "/" + str(len(self.devset)) + " processing file " + file)
            sys.stdout.flush()
            file_index += 1
            lab_file = file + ".lab"
            dio = DatasetIO(self.features)
            print ("\n"+self.setup_trainer.dev_data_folder)
            feats = dio.read_input_feats(self.setup_trainer.dev_data_folder+'/'+lab_file, self.features)




            import time
            start = time.time()
            mgc, att = self.vocoder.generate(feats, max_size=max_size)

            self.array2file(mgc, self.setup_trainer.train_output_folder+'/' + file[file.rfind('/') + 1:] + '.mgc')
            att = [a.value() for a in att]
            new_att = np.zeros((len(att), len(feats[0]), 3), dtype=np.uint8)

            for ii in range(len(feats[0]) ):
                for jj in range(len(att)):
                    val = np.clip(int(att[jj][ii] * 255), 0, 255)
                    new_att[jj, ii, 0] = val
                    new_att[jj, ii, 1] = val
                    new_att[jj, ii, 2] = val

            from PIL import Image
            img = Image.fromarray(new_att, 'RGB')
            img.save(self.setup_trainer.train_output_folder+'/' + file[file.rfind('/') + 1:] + 'att.png')

            output_file = self.setup_trainer.train_output_folder+'/' + file[file.rfind('/') + 1:] + '.png'
            bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
            for x in range(mgc.shape[0]):
                for y in range(mgc.shape[1]):
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
            #print mgc.shape
            output_file = self.setup_trainer.train_output_folder+'/' + file[file.rfind('/') + 1:] + '.png'
            bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
            for x in range(mgc.shape[0]):
                for y in range(mgc.shape[1]):
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
        dio = DatasetIO(self.features)

        if params.no_bounds:
            max_mgc = -1
        else:
            max_mgc = 1000

        self.synth_devset(max_size=max_mgc)
        self.vocoder.store(self.setup_trainer.models_folder+'/rnn_encoder')

        while left_itt > 0:
            sys.stdout.write("Starting epoch " + str(epoch) + "\n")
            sys.stdout.write("Shuffling training data\n")
            from random import shuffle

            shuffle(self.trainset)
            file_index = 1
            total_loss = 0
            for file in self.trainset:
                sys.stdout.write(
                    "\t" + str(file_index) + "/" + str(len(self.trainset)) + " processing file " + file)
                sys.stdout.flush()

                mgc_file = self.setup_trainer.train_data_folder+'/'+file + ".mgc.npy"
                mgc = np.load(mgc_file)

                lab_file = self.setup_trainer.train_data_folder+'/'+ file + ".lab"
                feats = dio.read_input_feats(lab_file, self.features)
                #print (feats)
                #sys.exit(1)
                file_index += 1

                import time
                start = time.time()

                if len(mgc) < 1400:
                    loss = self.vocoder.learn(feats, mgc, guided_att=not params.no_guided_attention)
                else:
                    sys.stdout.write(' too long, skipping')
                    loss = 0
                total_loss += loss
                stop = time.time()
                sys.stdout.write(' avg loss=' + str(loss) + " execution time=" + str(stop - start))
                sys.stdout.write('\n')
                sys.stdout.flush()
                if file_index % 200 == 0:
                    self.synth_devset(max_size=max_mgc)
                    self.vocoder.store(self.setup_trainer.models_folder+'/rnn_encoder')

            self.synth_devset(max_size=max_mgc)
            self.vocoder.store(self.setup_trainer.models_folder+'/rnn_encoder')

            epoch += 1
