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
        self.trainer = dy.AdamTrainer(self.model, alpha=2e-3, beta_1=0.9, beta_2=0.9)
        self.encodings = encodings

        self.DECODER_SIZE = 100
        self.ENCODER_SIZE = 100
        self.CHAR_EMB_SIZE = 100
        self.HIDDEN_SIZE = 100
        self.lexicon = {}

        self.char_lookup = self.model.add_lookup_parameters((len(self.encodings.char2int), self.CHAR_EMB_SIZE))
        self.phoneme_lookup = self.model.add_lookup_parameters(
            (len(self.encodings.phoneme2int) + 1, self.CHAR_EMB_SIZE))  # +1 is for special START

        self.start_lookup = self.model.add_lookup_parameters(
            (1, self.CHAR_EMB_SIZE + self.ENCODER_SIZE * 2))  # START SYMBOL

        self.encoder_fw = []
        self.encoder_bw = []

        input_layer_size = self.CHAR_EMB_SIZE
        for ii in range(2):
            self.encoder_fw.append(dy.VanillaLSTMBuilder(1, input_layer_size, self.ENCODER_SIZE, self.model))
            self.encoder_bw.append(dy.VanillaLSTMBuilder(1, input_layer_size, self.ENCODER_SIZE, self.model))

            input_layer_size = self.ENCODER_SIZE * 2

        self.decoder = dy.VanillaLSTMBuilder(2, self.ENCODER_SIZE * 2 + self.CHAR_EMB_SIZE, self.DECODER_SIZE,
                                             self.model)

        self.att_w1 = self.model.add_parameters((100, self.ENCODER_SIZE * 2))
        self.att_w2 = self.model.add_parameters((100, self.DECODER_SIZE))
        self.att_v = self.model.add_parameters((1, 100))

        self.hidden_w = self.model.add_parameters((self.HIDDEN_SIZE, self.DECODER_SIZE))
        self.hidden_b = self.model.add_parameters((self.HIDDEN_SIZE))

        self.softmax_w = self.model.add_parameters(
            (len(self.encodings.phoneme2int) + 1, self.HIDDEN_SIZE))  # +1 is for EOS
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

        return output_vectors, attention_weights

    def _make_input(self, word):
        emb_list = []
        for char in word:
            if char in self.encodings.char2int:
                emb_list.append(self.char_lookup[self.encodings.char2int[char]])
            else:
                emb_list.append(self.char_lookup[self.encodings.char2int['<UNK>']])

        return emb_list

    def _compute_guided_attention(self, att_vect, decoder_step, input_size, output_size):
        if output_size <= 1 or input_size <= 1:
            return dy.scalarInput(0)

        target_probs = []

        t1 = float(decoder_step) / output_size

        for encoder_step in range(input_size):
            target_probs.append(1.0 - np.exp(-((float(encoder_step) / input_size - t1) ** 2) / 0.08))

        # print target_probs
        target_probs = dy.inputVector(target_probs)
        # print (target_probs.npvalue().shape, att_vect.npvalue().shape)

        return dy.transpose(target_probs) * att_vect

    def _compute_binary_divergence(self, pred, target):
        return dy.binary_log_loss(pred, target)

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
        attention_list = []
        last_output = self.phoneme_lookup[len(self.encodings.phoneme2int)]
        phon_index = 0
        if not runtime:  # some nice dropouts
            zero_att = dy.inputVector([0 for ii in range(self.ENCODER_SIZE * 2)])
            zero_prev = dy.inputVector([0 for ii in range(self.CHAR_EMB_SIZE)])

        while True:
            att, att_weights = self._attend(encoder_vectors, decoder_state)
            attention_list.append(att_weights)

            s1 = 1
            s2 = 1
            if not runtime:
                r1 = np.random.random()
                r2 = np.random.random()
                if r1 < 0.34:
                    s2 = 2
                    att = zero_att
                if r2 < 0.34:
                    last_output = zero_prev
                    s1 = 1

            decoder_state = decoder_state.add_input(dy.concatenate([att * s1, last_output * s2]))
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
        return output_list, attention_list

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
        output_list, att_list = self._predict(word, gs_phones=transcription)

        for tp, pp, att, pos in zip(transcription, output_list, att_list, range(len(att_list))):
            self.losses.append(dy.pickneglogsoftmax(pp, self.encodings.phoneme2int[tp]))
            self.losses.append(self._compute_guided_attention(att, pos, len(word), len(transcription) + 1))

        self.losses.append(dy.pickneglogsoftmax(output_list[-1], len(self.encodings.phoneme2int)))

    def transcribe(self, word):
        dy.renew_cg()
        output, ignore = self._predict(word)
        transcription = [self.encodings.phoneme_list[np.argmax(value.npvalue())] for value in output]
        # print (word, transcription)
        return transcription

    def save(self, output_base):
        sys.stdout.write('\tStoring ' + output_base + '\n')
        self.model.save(output_base)

    def load(self, output_base):
        sys.stdout.write('\tLoading ' + output_base + '\n')
        self.model.populate(output_base)

    def load_lexicon(self, path):
        sys.stdout.write('\tLoading ' + path + '\n')

