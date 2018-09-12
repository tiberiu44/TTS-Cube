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


def med(l1, l2):
    mat = np.zeros((len(l1) + 1, len(l2) + 1))

    for ii in range(len(l1) + 1):
        mat[ii, 0] = ii

    for ii in range(len(l2) + 1):
        mat[0, ii] = ii

    for ii in range(1, len(l1) + 1):
        for jj in range(1, len(l2) + 1):
            if l1[ii - 1] != l2[jj - 1]:
                cost = 1
            else:
                cost = 0
            m = min([mat[ii - 1, jj - 1], mat[ii - 1, jj], mat[ii, jj - 1]])
            mat[ii, jj] = m + cost

    return mat[len(l1), len(l2)]


class G2PTrainer:

    def evaluate(self, model, dataset):
        total_edit_distance = 0
        total_chars = 0
        errors = 0
        for entry in dataset.entries:
            gold_phon = entry.transcription
            pred_phon = model.transcribe(entry.word)
            ed = med(gold_phon, pred_phon)
            total_edit_distance += ed
            if ed != 0:
                errors += 1
                # print (entry.word, gold_phon, pred_phon)

            total_chars += max([len(gold_phon), len(pred_phon)])

        return 1.0 - float(errors) / len(dataset.entries), total_edit_distance / total_chars

    def start_training(self, model, encodings, trainset, devset, output_base, batch_size=100, patience=20):
        encodings.save(output_base + '.encodings')

        itt_left = patience
        epoch = 0
        best_w_acc = 0
        best_p_acc = 0
        while itt_left > 0:
            epoch += 1
            itt_left -= 1

            sys.stdout.write('Starting epoch ' + str(epoch) + '\n')
            sys.stdout.write('\ttraining...')
            sys.stdout.flush()

            current_batch_size = 0
            model.start_batch()
            total_loss = 0
            index = 0
            last_proc = 0

            for entry in trainset.entries:
                index += 1
                curr_proc = int(index * 100 / len(trainset.entries))
                if curr_proc % 5 == 0 and curr_proc != last_proc:
                    while last_proc < curr_proc:
                        last_proc += 5
                        sys.stdout.write(' ' + str(last_proc))
                        sys.stdout.flush()

                current_batch_size += 1
                model.learn(entry.word, entry.transcription)
                if current_batch_size == batch_size:
                    total_loss += model.end_batch()
                    current_batch_size = 0

            if current_batch_size != 0:
                total_loss += model.end_batch()

            sys.stdout.write(' avg loss=' + str(total_loss / len(trainset.entries)) + '\n')
            sys.stdout.write('\tevaluating...')
            sys.stdout.flush()
            w_acc, p_acc = self.evaluate(model, devset)
            sys.stdout.write(' word accuracy=' + str(w_acc) + ' and phone edit distance=' + str(p_acc) + '\n')
            if w_acc > best_w_acc:
                model.save(output_base + '-bestAcc.network')
                best_w_acc = w_acc
                best_p_acc = p_acc

            model.save(output_base + '-last.network')

        sys.stdout.write(
            'Done with best word accuracy ' + str(best_w_acc) + 'and best phoneme edit distance ' + str(
                best_p_acc) + '\n')
