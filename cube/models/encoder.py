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


class Encoder:
    def __init__(self, params, num_phones, phone2int, model=None):
        self.model = model
        self.params = params
        self.PHONE_EMBEDDINGS_SIZE = 100
        self.ENCODER_SIZE = 200
        self.ENCODER_LAYERS = 2
        self.DECODER_SIZE = 200
        self.DECODER_LAYERS = 2
        self.phone2int = phone2int

        if self.model is None:
            self.model = dy.Model()
            self.trainer = dy.AdamTrainer(self.model)
            self.trainer.set_sparse_updates(True)
            self.trainer.set_clip_threshold(5.0)

        self.phone_lookup = self.model.add_lookup_parameters((num_phones + 2, self.PHONE_EMBEDDINGS_SIZE))
        from utils import orthonormal_VanillaLSTMBuilder
        self.encoder_fw = []
        self.encoder_bw = []
        self.encoder_fw.append(
            orthonormal_VanillaLSTMBuilder(1, self.PHONE_EMBEDDINGS_SIZE, self.ENCODER_SIZE, self.model))
        self.encoder_bw.append(
            orthonormal_VanillaLSTMBuilder(1, self.PHONE_EMBEDDINGS_SIZE, self.ENCODER_SIZE, self.model))

        for zz in xrange(1, self.ENCODER_LAYERS):
            self.encoder_fw.append(
                orthonormal_VanillaLSTMBuilder(1, self.ENCODER_SIZE * 2, self.ENCODER_SIZE, self.model))
            self.encoder_bw.append(
                orthonormal_VanillaLSTMBuilder(1, self.ENCODER_SIZE * 2, self.ENCODER_SIZE, self.model))

        self.decoder = orthonormal_VanillaLSTMBuilder(self.DECODER_LAYERS, self.ENCODER_SIZE * 2 + 100,
                                             self.DECODER_SIZE, self.model)

        # self.aux_hid_w = self.model.add_parameters((500, self.ENCODER_SIZE * 2))
        # self.aux_hid_b = self.model.add_parameters((500))
        # self.aux_proj_w = self.model.add_parameters((params.mgc_order, 500))
        # self.aux_proj_b = self.model.add_parameters((params.mgc_order))

        self.hid_w = self.model.add_parameters((500, self.DECODER_SIZE))
        self.hid_b = self.model.add_parameters((500))

        self.proj_w_1 = self.model.add_parameters((params.mgc_order, 500))
        self.proj_b_1 = self.model.add_parameters((params.mgc_order))
        self.proj_w_2 = self.model.add_parameters((params.mgc_order, 500))
        self.proj_b_2 = self.model.add_parameters((params.mgc_order))
        self.proj_w_3 = self.model.add_parameters((params.mgc_order, 500))
        self.proj_b_3 = self.model.add_parameters((params.mgc_order))

        self.highway_w = self.model.add_parameters((params.mgc_order, self.ENCODER_SIZE * 2))

        self.last_mgc_proj_w = self.model.add_parameters((100, self.params.mgc_order))
        self.last_mgc_proj_b = self.model.add_parameters((100))
        # self.last_att_proj_w = self.model.add_parameters((200, self.ENCODER_SIZE * 2))
        # self.last_att_proj_b = self.model.add_parameters((200))

        self.stop_w = self.model.add_parameters((1, self.DECODER_SIZE))
        self.stop_b = self.model.add_parameters((1))

        self.att_w1 = self.model.add_parameters((100, self.ENCODER_SIZE * 2))
        self.att_w2 = self.model.add_parameters((100, self.DECODER_SIZE))
        self.att_v = self.model.add_parameters((1, 100))

        self.start_lookup = self.model.add_lookup_parameters((1, params.mgc_order))
        self.decoder_start_lookup = self.model.add_lookup_parameters((1, self.ENCODER_SIZE * 2 + 100))

    def _make_input(self, characters):
        x_list = [self.phone_lookup[len(self.phone2int)]]
        for char in characters:
            if char in self.phone2int:
                x_list.append(self.phone_lookup[self.phone2int[char]])
        x_list.append(self.phone_lookup[len(self.phone2int) + 1])
        return x_list

    def _predict(self, characters, gold_mgc=None, max_size=-1):
        if gold_mgc is None:
            runtime = True
        else:
            runtime = False

        mgc_index = 0
        output_mgc = []
        output_stop = []
        # aux_output_mgc = []
        output_att = []
        last_mgc = self.start_lookup[0]

        # encoder
        x_input = self._make_input(characters)
        for lstm_fw, lstm_bw in zip(self.encoder_fw, self.encoder_bw):
            x_fw = lstm_fw.initial_state().transduce(x_input)
            x_bw = lstm_bw.initial_state().transduce(reversed(x_input))
            x_input = [dy.concatenate([fw, bw]) for fw, bw in zip(x_fw, reversed(x_bw))]

        encoder = x_input

        decoder = self.decoder.initial_state().add_input(self.decoder_start_lookup[0])
        last_att_pos = None
        if gold_mgc is None:
            last_att_pos = 0
        while True:
            att, align = self._attend(encoder, decoder, last_att_pos)

            if gold_mgc is None:
                last_att_pos = np.argmax(align.value())
            output_att.append(align)
            # main output
            mgc_proj = dy.tanh(self.last_mgc_proj_w.expr() * last_mgc + self.last_mgc_proj_b.expr())
            decoder = decoder.add_input(dy.concatenate([mgc_proj, att]))
            hidden = dy.tanh(self.hid_w.expr() * decoder.output() + self.hid_b.expr())

            output = dy.logistic(self.highway_w.expr() * att + self.proj_w_1.expr() * hidden + self.proj_b_1.expr())
            output_mgc.append(output)
            output = dy.logistic(self.highway_w.expr() * att + self.proj_w_2.expr() * hidden + self.proj_b_2.expr())
            output_mgc.append(output)
            output = dy.logistic(self.highway_w.expr() * att + self.proj_w_3.expr() * hidden + self.proj_b_3.expr())
            output_mgc.append(output)

            output_stop.append(dy.tanh(self.stop_w.expr() * decoder.output() + self.stop_b.expr()))

            if runtime:
                if max_size != -1 and mgc_index > max_size:
                    break
                last_mgc = dy.inputVector(output.value())
                if max_size == -1 and output_stop[-1].value < -0.5:
                    break

                if mgc_index>=len(characters)*10:#safeguard
                    break
            else:
                last_mgc = dy.inputVector(gold_mgc[min(mgc_index + 2, len(gold_mgc) - 1)])

            mgc_index += 3
            if not runtime and mgc_index >= gold_mgc.shape[0]:
                break
        return output_mgc, output_stop, output_att

    def _compute_guided_attention(self, att_vect, decoder_step, num_characters, num_mgcs):

        target_probs = []

        t1 = float(decoder_step) / num_mgcs

        for encoder_step in range(num_characters):
            target_probs.append(1.0 - np.exp(-((float(encoder_step) / num_characters - t1) ** 2) / 0.08))

        # print target_probs
        target_probs = dy.inputVector(target_probs)

        return dy.transpose(target_probs) * att_vect

    def _compute_binary_divergence(self, pred, target):
        return dy.binary_log_loss(pred, target)

    def learn(self, characters, target_mgc, guided_att=True):
        num_mgc = target_mgc.shape[0]
        # print num_mgc
        dy.renew_cg()
        output_mgc, output_stop, output_attention = self._predict(characters, target_mgc)
        losses = []
        index = 0
        for mgc, real_mgc in zip(output_mgc, target_mgc):
            t_mgc = dy.inputVector(real_mgc)
            # losses.append(self._compute_binary_divergence(mgc, t_mgc) )
            losses.append(dy.l1_distance(mgc, t_mgc))

            if index % 3 == 0:
                # attention loss
                if guided_att:
                    att = output_attention[index / 3]
                    losses.append(self._compute_guided_attention(att, index / 3, len(characters) + 2, num_mgc / 3))
                # EOS loss
                stop = output_stop[index / 3]
                if index >= num_mgc:
                    losses.append(dy.l1_distance(stop, dy.scalarInput(-1.0)))
                else:
                    losses.append(dy.l1_distance(stop, dy.scalarInput(1.0)))
            index += 1
        loss = dy.esum(losses)
        loss_val = loss.value() / num_mgc
        loss.backward()
        self.trainer.update()
        return loss_val

    def generate(self, characters, max_size=-1):
        dy.renew_cg()
        output_mgc, ignore1, att = self._predict(characters, max_size=max_size)
        mgc_output = [mgc.npvalue() for mgc in output_mgc]
        import numpy as np
        mgc_final = np.zeros((len(mgc_output), mgc_output[-1].shape[0]))
        for i in xrange(len(mgc_output)):
            for j in xrange(mgc_output[-1].shape[0]):
                mgc_final[i, j] = mgc_output[i][j]
        return mgc_final, att

    def store(self, output_base):
        self.model.save(output_base + ".network")

    def load(self, output_base):
        self.model.populate(output_base + ".network")

    def _attend(self, input_list, decoder_state, last_pos=None):
        w1 = self.att_w1.expr()
        w2 = self.att_w2.expr()
        v = self.att_v.expr()
        attention_weights = []

        w2dt = w2 * dy.concatenate([decoder_state.s()[-1]])
        for input_vector in input_list:
            attention_weight = v * dy.tanh(w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)

        attention_weights = dy.softmax(dy.concatenate(attention_weights))
        # force incremental attention if this is runtime
        if last_pos is not None:
            current_pos = np.argmax(attention_weights.value())
            if current_pos < last_pos or current_pos >= last_pos + 3:
                current_pos = last_pos + 1
                if current_pos >= len(input_list):
                    current_pos = len(input_list) - 1
                output_vectors = input_list[current_pos]
                simulated_att = np.zeros((len(input_list)))
                simulated_att[current_pos] = 1.0
                new_att_vec = dy.inputVector(simulated_att)
                return output_vectors, new_att_vec

        output_vectors = dy.esum(
            [vector * attention_weight for vector, attention_weight in zip(input_list, attention_weights)])

        return output_vectors, attention_weights
