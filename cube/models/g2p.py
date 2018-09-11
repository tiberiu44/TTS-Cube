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

import dynet as dy
import numpy as np


class G2P:
    def __init__(self, encodings):
        self.losses = []
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        self.encodings = encodings

        DECODER_SIZE = 100
        ENCODER_SIZE = 100
        CHAR_EMB_SIZE = 100
        HIDDEN_SIZE = 100

        self.char_lookup = self.model.add_lookup_parameters((len(self.encodings.char2int), CHAR_EMB_SIZE))

        self.encoder_fw = []
        self.encoder_bw = []

        input_layer_size = CHAR_EMB_SIZE
        for ii in range(2):
            self.encoder_fw.append(dy.VanillaLSTMBuilder(1, ENCODER_SIZE, input_layer_size, self.model))
            self.encoder_bw.append(dy.VanillaLSTMBuilder(1, ENCODER_SIZE, input_layer_size, self.model))
            input_layer_size = ENCODER_SIZE * 2

        self.decoder = dy.VanillaLSTMBuilder(2, DECODER_SIZE, ENCODER_SIZE * 2, self.model)

        self.att_w1 = self.model.add_parameters((100, ENCODER_SIZE * 2))
        self.att_w2 = self.model.add_parameters((100, DECODER_SIZE))
        self.att_v = self.model.add_parameters((1, 100))

        self.hidden_w = self.model.add_parameters((HIDDEN_SIZE, DECODER_SIZE))
        self.hidden_b = self.model.add_parameters((HIDDEN_SIZE))

        self.softmax_w = self.model.add_parameters((len(self.encodings.phoneme2int), HIDDEN_SIZE))
        self.softmax_b = self.model.add_parameters((len(self.encodings.phoneme2int)))

    def _attend(self, input_vectors, decoder):
        w1 = self.att_w1.expr(update=True)
        w2 = self.att_w2.expr(update=True)
        v = self.att_v.expr(update=True)
        attention_weights = []

        w2dt = w2 * decoder.s()[-1]
        for input_vector in input_vectors:
            attention_weight = v * dy.tanh(w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)

        attention_weights = dy.softmax(dy.concatenate(attention_weights))

        output_vectors = dy.esum(
            [vector * attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])

        return output_vectors

    def _predict(self, word, gs_phones=None):
        return []

    def start_batch(self):
        self.losses = []

    def end_batch(self):
        total_loss = 0
        if len(self.losses) != 0:
            loss = dy.esum(self.losses)
            self.losses = []
            total_loss = loss.value()
            loss.backward()
            self.trainer.update()
        return total_loss

    def learn(self, word, transcription):
        return None

    def transcribe(self, word):
        output = self._predict(word)
        transcription = [np.argmax(value.npvalue()) for value in output]
        return transcription
