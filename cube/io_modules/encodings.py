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

class Encodings:
    def __init__(self):
        self.phoneme_list = []
        self.phoneme2int = {}
        self.char2int = {}
        self.char2int['<UNK>'] = 0

    def update_encodings(self, dataset):
        char_count = {}
        for entry in dataset.entries:
            for char in entry.word:
                if char not in char_count:
                    char_count[char] = 1
                else:
                    char_count[char] += 1

            for phoneme in entry.transcription:
                if phoneme not in self.phoneme2int:
                    self.phoneme2int[phoneme] = len(self.phoneme2int)
                    self.phoneme_list.append(phoneme)

        for char in char_count:
            if char_count[char] >= 2:
                self.char2int[char] = len(self.char2int)

    def save(self, filename):
        f = open(filename, "w")
        f.write('CHARS ' + str(len(self.char2int)) + '\n')
        for char in self.char2int:
            f.write(char + '\t' + str(self.char2int[char]) + '\n')

        f.write('PHONEMES ' + str(len(self.phoneme2int)) + '\n')
        for phon in self.phoneme2int:
            f.write(phon + '\t' + str(self.phoneme2int[phon]) + '\n')

        f.close()

    def load(self, filename):
        f = open(filename)
        line = f.readline()
        parts = line.split(' ')
        num_chars = int(parts[1])
        for ii in range(num_chars):
            line = f.readline()
            parts = line.split('\t')
            char = parts[0]
            index = int(parts[1])
            self.char2int[char] = index

        line = f.readline()
        parts = line.split(' ')
        num_phones = int(parts[1])
        self.phoneme_list = ["" for ii in range(num_phones)]
        for ii in range(num_phones):
            line = f.readline()
            parts = line.split('\t')
            phoneme = parts[0]
            index = int(parts[1])
            self.phoneme2int[phoneme] = index
            self.phoneme_list[index] = phoneme
        f.close()
