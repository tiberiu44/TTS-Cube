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
import sys


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
        self.phoneme_lookup = self.model.add_lookup_parameters(
            (len(self.encodings.phoneme2int) + 1, CHAR_EMB_SIZE))  # +1 is for special START

        self.start_lookup = self.model.add_lookup_parameters((1, CHAR_EMB_SIZE + ENCODER_SIZE * 2))  # START SYMBOL

        self.encoder_fw = []
        self.encoder_bw = []

        input_layer_size = CHAR_EMB_SIZE
        for ii in range(2):
            self.encoder_fw.append(dy.VanillaLSTMBuilder(1, input_layer_size, ENCODER_SIZE, self.model))
            self.encoder_bw.append(dy.VanillaLSTMBuilder(1, input_layer_size, ENCODER_SIZE, self.model))

            input_layer_size = ENCODER_SIZE * 2

        self.decoder = dy.VanillaLSTMBuilder(2, ENCODER_SIZE * 2 + CHAR_EMB_SIZE, DECODER_SIZE, self.model)

        self.att_w1 = self.model.add_parameters((100, ENCODER_SIZE * 2))
        self.att_w2 = self.model.add_parameters((100, DECODER_SIZE))
        self.att_v = self.model.add_parameters((1, 100))

        self.hidden_w = self.model.add_parameters((HIDDEN_SIZE, DECODER_SIZE))
        self.hidden_b = self.model.add_parameters((HIDDEN_SIZE))

        self.softmax_w = self.model.add_parameters((len(self.encodings.phoneme2int) + 1, HIDDEN_SIZE))  # +1 is for EOS
        self.softmax_b = self.model.add_parameters((len(self.encodings.phoneme2int) + 1))

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

    def _make_input(self, word):
        emb_list = []
        for char in word:
            if char in self.encodings.char2int:
                emb_list.append(self.char_lookup[self.encodings.char2int[char]])
            else:
                emb_list.append(self.char_lookup[self.encodings.char2int['<UNK>']])

        return emb_list

    def _predict(self, word, gs_phones=None):
        if gs_phones is None:
            runtime = True
        else:
            runtime = False

        input = self._make_input(word)
        input_list = input

        for fw, bw in zip(self.encoder_fw, self.encoder_bw):
            fw_state = fw.initial_state()
            bw_state = bw.initial_state()
            fw_out = fw_state.transduce(input_list)
            bw_out = list(reversed(bw_state.transduce(reversed(input_list))))
            input_list = [dy.concatenate([f, b]) for f, b in zip(fw_out, bw_out)]

        encoder_vectors = input_list

        decoder_state = self.decoder.initial_state().add_input(self.start_lookup[0])
        output_list = []
        last_output = self.phoneme_lookup[len(self.encodings.phoneme2int)]
        phon_index = 0
        while True:
            att = self._attend(encoder_vectors, decoder_state)
            decoder_state = decoder_state.add_input(dy.concatenate([att, last_output]))
            hidden = dy.tanh(
                self.hidden_w.expr(update=True) * decoder_state.output() + self.hidden_b.expr(update=True))
            softmax = self.softmax_w.expr(update=True) * hidden + self.softmax_b.expr(update=True)
            if runtime:
                softmax = dy.softmax(softmax)
                sel_index = np.argmax(softmax.npvalue())
                last_output = self.phoneme_lookup[sel_index]
                if sel_index == len(self.encodings.phoneme2int):
                    break  # EOS

                if phon_index > 255: break  # extra exit condition
            else:
                if phon_index < len(gs_phones):
                    last_phoneme = gs_phones[phon_index]
                    last_output = self.phoneme_lookup[self.encodings.phoneme2int[last_phoneme]]
                else:
                    output_list.append(softmax)
                    break  # EOS

            phon_index += 1
            output_list.append(softmax)
        return output_list

    def start_batch(self):
        self.losses = []
        dy.renew_cg()

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
        output_list = self._predict(word, gs_phones=transcription)
        for tp, pp in zip(transcription, output_list):
            self.losses.append(dy.pickneglogsoftmax(pp, self.encodings.phoneme2int[tp]))

        self.losses.append(dy.pickneglogsoftmax(pp, len(self.encodings.phoneme2int)))

    def transcribe(self, word):
        dy.renew_cg()
        output = self._predict(word)
        transcription = [self.encodings.phoneme_list[np.argmax(value.npvalue())] for value in output]
        #print (word, transcription)
        return transcription

    def save(self, output_base):
        sys.stdout.write('\tStoring ' + output_base + '\n')
        self.model.save(output_base)

    def load(self, output_base):
        sys.stdout.write('\tLoading ' + output_base + '\n')
        self.model.populate(output_base)
