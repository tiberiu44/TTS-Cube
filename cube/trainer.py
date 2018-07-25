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

import dynet_config
import optparse
import sys
import numpy as np

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--cleanup', action='store_true', dest='cleanup',
                      help='Cleanup temporary training files and start from fresh')
    parser.add_option('--phase', action='store', dest='phase',
                      choices=['1', '2', '3', '4', '5'],
                      help='select phase: 1 - prepare corpus; 2 - train vocoder; 3 - train encoder; 4 - end-to-end; 5 - test vocoder')
    parser.add_option("--batch-size", action='store', dest='batch_size', default='1000', type='int',
                      help='number of samples in a single batch (default=1000)')
    parser.add_option("--set-mem", action='store', dest='memory', default='2048', type='int',
                      help='preallocate memory for batch training (default 2048)')
    parser.add_option("--autobatch", action='store_true', dest='autobatch',
                      help='turn on/off dynet autobatching')
    parser.add_option("--resume", action='store_true', dest='resume',
                      help='resume from last checkpoint')
    parser.add_option("--no-guided-attention", action='store_true', dest='no_guided_attention',
                      help='disable guided attention')
    parser.add_option("--no-bounds", action='store_true', dest='no_bounds',
                      help='disable fixed synthesis length')
    parser.add_option("--use-gpu", action='store_true', dest='gpu',
                      help='turn on/off GPU support')
    parser.add_option('--train-folder', action='store', dest='train_folder',
                      help='Location of the training files')
    parser.add_option('--dev-folder', action='store', dest='dev_folder',
                      help='Location of the development files')
    parser.add_option('--target-sample-rate', action='store', dest='target_sample_rate',
                      help='Resample input files at this rate (default=16000)', type='int', default=16000)
    parser.add_option('--mgc-order', action='store', dest='mgc_order', type='int',
                      help='Order of MGC parameters (default=80)', default=60)


    (params, _) = parser.parse_args(sys.argv)

    memory = int(params.memory)
    if params.autobatch:
        autobatch = True
    else:
        autobatch = False
    dynet_config.set(mem=memory, random_seed=9, autobatch=autobatch)
    if params.gpu:
        dynet_config.set_gpu()


    def array2file(a, filename):
        np.save(filename, a)


    def file2array(filename):
        a = np.load(filename)
        return a


    def render_spectrogram(mgc, output_file):
        bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
        mgc_min = mgc.min()
        mgc_max = mgc.max()

        for x in xrange(mgc.shape[0]):
            for y in xrange(mgc.shape[1]):
                val = (mgc[x, y] - mgc_min) / (mgc_max - mgc_min)

                color = val * 255
                bitmap[mgc.shape[1] - y - 1, x] = [color, color, color]
        import scipy.misc as smp

        img = smp.toimage(bitmap)
        img.save(output_file)


    def phase_1_prepare_corpus(params):
        from os import listdir
        from os.path import isfile, join
        from os.path import exists
        train_files_tmp = [f for f in listdir(params.train_folder) if isfile(join(params.train_folder, f))]
        dev_files_tmp = [f for f in listdir(params.dev_folder) if isfile(join(params.dev_folder, f))]

        sys.stdout.write("Scanning training files...")
        sys.stdout.flush()
        final_list = []
        for file in train_files_tmp:
            base_name = file[:-4]
            lab_name = base_name + '.txt'
            wav_name = base_name + '.wav'
            if exists(join(params.train_folder, lab_name)) and exists(join(params.train_folder, wav_name)):
                if base_name not in final_list:
                    final_list.append(base_name)

        train_files = final_list
        sys.stdout.write(" found " + str(len(train_files)) + " valid training files\n")
        sys.stdout.write("Scanning development files...")
        sys.stdout.flush()
        final_list = []
        for file in dev_files_tmp:
            base_name = file[:-4]
            lab_name = base_name + '.txt'
            wav_name = base_name + '.wav'
            if exists(join(params.dev_folder, lab_name)) and exists(join(params.dev_folder, wav_name)):
                if base_name not in final_list:
                    final_list.append(base_name)

        dev_files = final_list
        sys.stdout.write(" found " + str(len(dev_files)) + " valid development files\n")

        from io_modules.dataset import DatasetIO
        from io_modules.vocoder import MelVocoder
        from shutil import copyfile
        import pysptk
        dio = DatasetIO()
        vocoder = MelVocoder()
        base_folder = params.train_folder
        for index in xrange(len(train_files)):
            sys.stdout.write("\r\tprocessing file " + str(index + 1) + "/" + str(len(train_files)))
            sys.stdout.flush()
            base_name = train_files[index]
            lab_name = base_name + '.txt'
            wav_name = base_name + '.wav'
            spc_name = base_name + '.png'

            copyfile(join(base_folder, lab_name), join('data/processed/train', lab_name))

            # WAVE
            data, sample_rate = dio.read_wave(join(base_folder, wav_name), sample_rate=params.target_sample_rate)
            mgc = vocoder.melspectrogram(data, sample_rate=params.target_sample_rate, num_mels=params.mgc_order)
            # SPECT
            render_spectrogram(mgc, join('data/processed/train', spc_name))
            dio.write_wave(join('data/processed/train', base_name + '.orig.wav'), data, sample_rate)
            array2file(mgc, join('data/processed/train', base_name + '.mgc'))

        sys.stdout.write('\n')
        base_folder = params.dev_folder
        for index in xrange(len(dev_files)):
            sys.stdout.write("\r\tprocessing file " + str(index + 1) + "/" + str(len(dev_files)))
            sys.stdout.flush()
            base_name = dev_files[index]
            lab_name = base_name + '.txt'
            wav_name = base_name + '.wav'
            spc_name = base_name + '.png'

            copyfile(join(base_folder, lab_name), join('data/processed/dev/', lab_name))
            # WAVE
            data, sample_rate = dio.read_wave(join(base_folder, wav_name), sample_rate=params.target_sample_rate)
            mgc = vocoder.melspectrogram(data, sample_rate=params.target_sample_rate, num_mels=params.mgc_order)
            # SPECT
            render_spectrogram(mgc, join('data/processed/dev', spc_name))
            dio.write_wave(join('data/processed/dev', base_name + '.orig.wav'), data, sample_rate)
            array2file(mgc, join('data/processed/dev', base_name + '.mgc'))

        sys.stdout.write('\n')


    def phase_2_train_vocoder(params):
        from io_modules.dataset import Dataset
        from models.vocoder import Vocoder
        from trainers.vocoder import Trainer
        vocoder = Vocoder(params)
        if params.resume:
            sys.stdout.write('Resuming from previous checkpoint\n')
            vocoder.load('data/models/rnn')
        trainset = Dataset("data/processed/train")
        devset = Dataset("data/processed/dev")
        sys.stdout.write('Found ' + str(len(trainset.files)) + ' training files and ' + str(
            len(devset.files)) + ' development files\n')
        trainer = Trainer(vocoder, trainset, devset)
        trainer.start_training(20, params.batch_size, params.target_sample_rate)


    def phase_3_train_encoder(params):
        from io_modules.dataset import Dataset
        from models.encoder import Encoder
        from trainers.encoder import Trainer
        trainset = Dataset("data/processed/train")
        devset = Dataset("data/processed/dev")
        sys.stdout.write('Found ' + str(len(trainset.files)) + ' training files and ' + str(
            len(devset.files)) + ' development files\n')

        character2int = {}
        for train_file in trainset.files:
            from io_modules.dataset import DatasetIO
            dio = DatasetIO()
            lab_list = dio.read_lab(train_file + ".txt")
            for entry in lab_list:
                if entry.phoneme not in character2int:
                    character2int[entry.phoneme] = len(character2int)
        sys.stdout.write('Found ' + str(len(character2int)) + ' unique phonemes\n')

        f = open('data/models/encoder.chars', 'w')
        for char in character2int:
            f.write(char.encode('utf-8') + '\t' + str(character2int[char]) + '\n')
        f.close()

        encoder = Encoder(params, len(character2int), character2int)
        if params.resume:
            sys.stdout.write('Resuming from previous checkpoint\n')
            encoder.load('data/models/rnn_encoder')
        if params.no_guided_attention:
            sys.stdout.write('Disabling guided attention\n')
        if params.no_bounds:
            sys.stdout.write('Using internal stopping condition for synthesis\n')
        trainer = Trainer(encoder, trainset, devset)
        trainer.start_training(10, 1000, params)


    def phase_5_test_vocoder(params):
        from io_modules.dataset import Dataset
        from models.vocoder import Vocoder
        from trainers.vocoder import Trainer
        vocoder = Vocoder(params)
        vocoder.load('data/models/rnn')
        trainset = Dataset("data/processed/train")
        devset = Dataset("data/processed/dev")
        sys.stdout.write('Found ' + str(len(trainset.files)) + ' training files and ' + str(
            len(devset.files)) + ' development files\n')
        trainer = Trainer(vocoder, trainset, devset)
        trainer.synth_devset(params.batch_size, target_sample_rate=params.target_sample_rate, sample=True,
                             temperature=0.8)


    if params.phase and params.phase == '1':
        phase_1_prepare_corpus(params)
    if params.phase and params.phase == '2':
        phase_2_train_vocoder(params)
    if params.phase and params.phase == '3':
        phase_3_train_encoder(params)
    if params.phase and params.phase == '5':
        phase_5_test_vocoder(params)
