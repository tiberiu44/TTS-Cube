#
# Author: Adriana Stan, october 2018
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
import numpy as np
from os.path import exists
from os import makedirs



class Setup_Training:

    def __init__(self, train_folder, dev_folder, out_folder):
        self.train_folder = train_folder
        self.dev_folder = dev_folder
        self.out_folder = out_folder
        self.train_data_folder, self.dev_data_folder, self.model_folder = self.create_folder_struct()
        self.train_file_list = self.read_file_list(self.train_folder)
        self.dev_file_list = self.read_file_list(self.dev_folder)


    def setup(self):
        self.generate_labels(self.train_folder, self.train_data_folder, self.train_file_list)
        self.generate_labels(self.dev_folder, self.dev_data_folder, self.dev_file_list)
        self.generate_acoustic_params(self.train_folder, self.train_data_folder, self.train_file_list)
        self.generate_acoustic_params(self.dev_folder, self.dev_data_folder, self.dev_file_list)


    def array2file(self, a, filename):
        np.save(filename, a)


    def file2array(self, filename):
        a = np.load(filename)
        return a

    def create_folder_struct(self):
        train_data_folder = self.out_folder + "/data/processed/train/"
        dev_data_folder = self.out_folder + "/data/processed/dev/"
        models_folder = self.out_folder + "/data/models/"
        if not exists(train_data_folder):
            makedirs(train_data_folder)
        if not exists(dev_data_folder):
            makedirs(dev_data_folder)
        if not exists(models_folder):
            makedirs(models_folder)
        return train_data_folder, dev_data_folder, models_folder


    def read_file_list (self, folder):
        from os import listdir
        from os.path import isfile, join
        files_tmp = [f for f in listdir(folder) if isfile(join(folder, f))]
        final_list = []
        for file in files_tmp:
                base_name = file[:-4]
                lab_name = base_name + '.txt'
                wav_name = base_name + '.wav'
                if exists(join(folder, lab_name)) and exists(join(folder, wav_name)):
                    if base_name not in final_list:
                        final_list.append(base_name)
        return final_list




    def generate_labels(self, in_folder, out_folder, file_list):
        from os.path import  join
        import sys
        from shutil import copyfile

        for index in range(len(file_list)):
            sys.stdout.write("\r\tcreating label file " + str(index + 1) + "/" + str(len(file_list)))
            sys.stdout.flush()
            base_name = file_list[index]
            txt_name = base_name + '.txt'
            lab_name = base_name + '.lab'

            # LAB - copy or create
            #if exists(join(in_folder, lab_name)):
            #    copyfile(join(in_folder, lab_name), join(out_folder, lab_name))
            #else:
            #     create_lab_file(join(in_folder, txt_name), join(out_folder, lab_name))

            # TXT
            copyfile(join(in_folder, txt_name), join(out_folder, txt_name))
            copyfile(join(in_folder, txt_name), join(out_folder, lab_name))

        sys.stdout.write('\n')


    def generate_acoustic_params(self,in_folder, out_folder, file_list, sample_rate=16000, mgc_order=60):

        # WAVE
        from io_modules.vocoder import MelVocoder
        from io_modules.acoustic import Acoustic
        from os.path import join
        import sys

        vocoder = MelVocoder()
        ac = Acoustic()

        for index in range(len(file_list)):
            sys.stdout.write("\r\tgenerating acoustic features " + str(index + 1) + "/" + str(len(file_list)))
            sys.stdout.flush()
            base_name = file_list[index]
            wav_name = base_name + '.wav'
            spc_name = base_name + '.png'
            data, sample_rate = ac.read_wave(join(in_folder, wav_name), sample_rate=sample_rate)
            mgc = vocoder.melspectrogram(data, sample_rate=sample_rate, num_mels=mgc_order)
            # SPECT
            ac.render_spectrogram(mgc, join(out_folder, spc_name))
            ac.write_wave(join(out_folder, base_name + '.orig.wav'), data, sample_rate)
            self.array2file(mgc, join(out_folder, base_name + '.mgc'))

        sys.stdout.write('\n')




    '''
    def create_lab_file(txt_file, lab_file):
        fin = open(txt_file, 'r')
        fout = open(lab_file, 'w')
        line = fin.readline().strip().replace('\t', ' ')
        while True:
            nl = line.replace('  ', ' ')
            if nl == line:
                break
            line = nl
    
        fout.write('START\n')
        for char in line:
            l_char = char.lower()
            style = 'CASE:lower'
            if l_char == l_char.upper():
                style = 'CASE:symb'
            elif l_char != char:
                style = 'CASE:upper'
            if len(txt_file.replace('\\', '/').split('/')[-1].split('_')) != 1:
                speaker = 'SPEAKER:' + txt_file.replace('\\', '/').split('_')[0].split('/')[-1]
            else:
                speaker = 'SPEAKER:none'
            fout.write(l_char + '\t' + speaker + '\t' + style + '\n')
    
        fout.write('STOP\n')
    
        fin.close()
        fout.close()
        return ""
    '''