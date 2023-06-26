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
import optparse
import torch
import torch.nn as nn

sys.path.append('')
from cube.networks.modules import Seq2Seq


class G2P:
    def __init__(self):
        self.seq2seq = None
        self.token2int = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
        self.label2int = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
        self.label_list = ['<PAD>', '<UNK>', '<EOS>']
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.simple_tokenizer = SimpleTokenizer()
        self.lookup = {}

    def to(self, device):
        self.seq2seq.to(device)

    def load(self, path, load_last=False):
        import json
        f = open('{0}.encodings'.format(path), 'r')
        json_obj = json.load(f)
        self.token2int = json_obj['token2int']
        self.label2int = json_obj['label2int']
        self.label_list = json_obj['label_list']
        f.close()
        self.initialize_network()
        if load_last:
            self.seq2seq.load('{0}.last'.format(path))
        else:
            try:
                self.seq2seq.load('{0}.best'.format(path))
            except:
                self.seq2seq.load('{0}.model'.format(path))

    def save(self, path):
        f = open('{0}.encodings'.format(path), 'w')
        json_obj = {'token2int': self.token2int, 'label_list': self.label_list, 'label2int': self.label2int}
        import json
        json.dump(json_obj, f, indent=2)
        f.close()

    def update_encodings(self, dataset, cutoff=2):
        token2count = {}
        label2count = {}
        for example in dataset.examples:
            word = example[0].lower()
            trans = example[1]
            for char in word:
                if char not in token2count:
                    token2count[char] = 1
                else:
                    token2count[char] += 1
            for phon in trans:
                if phon not in label2count:
                    label2count[phon] = 1
                else:
                    label2count[phon] += 1

        for token in token2count:
            if token2count[token] >= cutoff:
                self.token2int[token] = len(self.token2int)
        for label in label2count:
            if label2count[label] >= cutoff:
                self.label2int[label] = len(self.label2int)
                self.label_list.append(label)

    def initialize_network(self):
        self.seq2seq = Seq2Seq(len(self.token2int), len(self.label2int))

    def learn_batch(self, batch):
        import numpy as np
        max_len_x = max([len(example[0]) for example in batch])
        max_len_y = max([len(example[1]) for example in batch])
        x = np.zeros((len(batch), max_len_x + 1))
        y = np.zeros((len(batch), max_len_y + 1))
        for ii in range(x.shape[0]):
            for jj in range(x.shape[1]):
                idx = self.token2int['<PAD>']
                if jj < len(batch[ii][0]):
                    idx = self.token2int['<UNK>']
                    if batch[ii][0][jj].lower() in self.token2int:
                        idx = self.token2int[batch[ii][0][jj].lower()]
                elif jj == len(batch[ii][0]):
                    idx = self.token2int['<EOS>']
                x[ii, jj] = idx

            for jj in range(y.shape[1]):
                idx = self.label2int['<PAD>']
                if jj < len(batch[ii][1]):
                    idx = self.label2int['<UNK>']
                    if batch[ii][1][jj] in self.label2int:
                        idx = self.label2int[batch[ii][1][jj]]
                elif jj == len(batch[ii][1]):
                    idx = self.label2int['<EOS>']
                y[ii, jj] = idx

        x = torch.tensor(x, device=self._get_device(), dtype=torch.long)
        y_target = torch.tensor(y, device=self._get_device(), dtype=torch.long)
        y_pred = self.seq2seq(x, gs_output=y_target)
        return self.criterion(y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], -1), y_target.view(-1))

    def transcribe(self, words):
        import numpy as np
        max_len_x = max([len(example) for example in words])
        x = np.zeros((len(words), max_len_x + 1))
        for ii in range(x.shape[0]):
            for jj in range(x.shape[1]):
                idx = self.token2int['<PAD>']
                if jj < len(words[ii]):
                    idx = self.token2int['<UNK>']
                    if words[ii][jj].lower() in self.token2int:
                        idx = self.token2int[words[ii][jj].lower()]
                elif jj == len(words[ii]):
                    idx = self.token2int['<EOS>']
                x[ii, jj] = idx
        x = torch.tensor(x, device=self._get_device(), dtype=torch.long)
        with torch.no_grad():
            pred_y = self.seq2seq(x)
        transcriptions = []
        # from ipdb import set_trace
        # set_trace()
        pred_y = pred_y.detach().cpu().numpy()
        for trans in pred_y:
            tr = []
            for ph in trans:
                index = np.argmax(ph)
                if index == self.label2int['<EOS>']:
                    break
                if index != self.label2int['<PAD>'] and index != self.label2int['<UNK>']:
                    tr.append(self.label_list[index])
            transcriptions.append(tr)

        return transcriptions

    def load_lexicon(self, path):
        lines = open(path).readlines()
        for line in lines:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) != 2:
                print(parts)
                continue
            word = parts[0].lower()
            transcription = parts[1].split(' ')
            self.lookup[word] = transcription

    def _get_device(self):
        if self.seq2seq.output_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self.seq2seq.output_emb.weight.device.type,
                                str(self.seq2seq.output_emb.weight.device.index))

    def __call__(self, utterance, trace=False):
        # print(utterance)
        tokens = self.simple_tokenizer(utterance)
        words = []
        trace_words = []
        for token in tokens:
            if token.is_word:
                words.append(token.word.lower())
            trace_words.append({'word': token.word})
        if len(words) == 0:
            transcriptions = []
        else:
            transcriptions = self.transcribe(words)
        i_trans = 0
        i_trace = 0
        for token in tokens:
            if token.is_word:
                token.transcription = transcriptions[i_trans]
                i_trans += 1
                if token.word.lower() in self.lookup:
                    token.transcription = self.lookup[token.word.lower()]
            else:
                if token.word == ' ':
                    token.transcription = [' ']  # [c for c in token.word]
                elif token.word == '-':
                    token.transcription = ['_']
                elif token.word == '"':
                    token.transcription = ['_']
                else:
                    token.transcription = ['']  # [c for c in token.word]
                # token.transcription = [c for c in token.word]
            trace_words[i_trace]['transcription'] = token.transcription
            i_trace += 1
        if not trace:
            return tokens
        else:
            return tokens, trace_words

    def evaluate(self, dataset):
        err = 0
        total = len(dataset.examples)
        batches = _get_batches(dataset.examples, batch_size=64)
        import tqdm
        for batch in tqdm.tqdm(batches, ncols=80, desc='\tEvaluating'):
            batch_x = [example[0] for example in batch]
            transcriptions = self.transcribe(batch_x)
            for ii in range(len(batch)):
                pred_tr = transcriptions[ii]
                gold_tr = batch[ii][1]
                if pred_tr != gold_tr:
                    err += 1
        return 1.0 - err / total

    def train(self):
        self.seq2seq.train()

    def eval(self):
        self.seq2seq.eval()


class Token:
    def __init__(self, word='', transcription=[], is_word=False):
        self.word = word
        self.transcription = transcription
        self.is_word = is_word

    def __repr__(self):
        if len(self.transcription) == 0:
            return '"{0}"'.format(self.word)
        else:
            return '{0}'.format(self.transcription)


class SimpleTokenizer:
    def __init__(self):
        pass

    def __call__(self, utterance):
        tokens = []
        cb = ''
        for char in utterance:
            if char.isalpha() or char == '\'':
                cb += char
            else:
                if cb != '':
                    tokens.append(Token(word=cb, is_word=True))
                    cb = ''
                tokens.append(Token(word=char, is_word=False))
        if cb != '':
            tokens.append(Token(word=cb, is_word=True))
        return tokens


class G2PDataset:
    def __init__(self, file):
        self.examples = []
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    print(parts)
                    continue

                word = parts[0]
                trans = parts[1].split(' ')
                self.examples.append((word, trans))


def _get_batches(examples, batch_size=16):
    batches = []
    cb = []
    for example in examples:
        cb.append(example)
        if len(cb) == batch_size:
            batches.append(cb)
            cb = []
    if len(batches) != 0:
        batches.append(cb)
    return batches


def _start_train(params):
    train = G2PDataset(params.train_file)
    dev = G2PDataset(params.dev_file)
    g2p = G2P()
    if not params.model_path:
        g2p.update_encodings(train)
        g2p.initialize_network()
        g2p.save(params.output_path)
        g2p.to(params.device)
        best_acc = 0
    else:
        g2p.load(params.model_path, load_last=True)
        g2p.to(params.device)
        g2p.seq2seq.eval()
        best_acc = g2p.evaluate(dev)
        sys.stdout.write('Setting baseline accuracy to {0:.4f}\n'.format(best_acc))
    optim = torch.optim.Adam(g2p.seq2seq.parameters(), lr=params.lr)

    patience_left = params.patience

    epoch = 1
    g2p.seq2seq.save('{0}.last'.format(params.output_path))
    while patience_left > 0:
        g2p.seq2seq.train()
        patience_left -= 1

        import random
        import tqdm

        sys.stdout.write('\n\nStarting epoch {0}\n'.format(epoch))
        random.shuffle(train.examples)
        batches = _get_batches(train.examples, batch_size=params.batch_size)
        total_loss = 0
        progb = tqdm.tqdm(batches, ncols=80, desc='\tloss=NaN')
        for batch in progb:
            loss = g2p.learn_batch(batch)
            total_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
            progb.set_description('\tloss={0:.6f}'.format(loss.item()))

        total_loss /= len(batches)
        sys.stdout.write('\tAvg loss: {0}\n'.format(total_loss))
        g2p.seq2seq.eval()
        acc = g2p.evaluate(dev)
        sys.stdout.write('\tDevset accuracy: {0}\n'.format(acc))
        if acc > best_acc:
            best_acc = acc
            sys.stdout.write('\tStoring {0}.best\n'.format(params.output_path))
            g2p.seq2seq.save('{0}.best'.format(params.output_path))
            patience_left = params.patience

        sys.stdout.write('\tStoring {0}.last\n'.format(params.output_path))
        g2p.seq2seq.save('{0}.last'.format(params.output_path))
        epoch += 1


def _eval(params):
    dev = G2PDataset(params.test_file)
    g2p = G2P()
    g2p.load(params.model_path)
    g2p.to(params.device)
    g2p.seq2seq.eval()
    acc = g2p.evaluate(dev)
    sys.stdout.write('Word accuracy rate is {0:.2f}%\n'.format(acc * 100))


def _transcribe(params):
    g2p = G2P()
    g2p.load(params.model_base)
    g2p.to(params.device)
    g2p.seq2seq.eval()
    f = open(params.output_file, 'w')
    lines = open(params.transcribe_file).readlines()
    BS = 128
    n_batch = len(lines) // BS
    if len(lines) % BS != 0:
        n_batch += 1

    for iBatch in range(n_batch):
        start = iBatch * BS
        stop = min(start + BS, len(lines))
        words = [p.split('\t')[0].strip() for p in lines[start:stop]]
        trans = g2p.transcribe(words)
        for w, t in zip(words, trans):
            f.write('{0}\t{1}\n'.format(w, ' '.join(t)))
    f.close()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--patience', action='store', dest='patience', default=20, type='int',
                      help='Num epochs without improvement (default=20)')
    parser.add_option('--train-file', action='store', dest='train_file',
                      help='Training file for g2p')
    parser.add_option('--dev-file', action='store', dest='dev_file',
                      help='Validation file for g2p')
    parser.add_option('--store', action='store', dest='output_path',
                      help='Base path for storing output model')
    parser.add_option("--batch-size", action='store', dest='batch_size', default='32', type='int',
                      help='number of samples in a single batch (default=32)')
    parser.add_option("--resume", action='store_true', dest='resume', help='Resume from previous checkpoint')
    parser.add_option("--device", action="store", dest="device", default='cuda:0')
    parser.add_option("--lr", action="store", dest="lr", default=1e-3, type=float)
    parser.add_option("--load", action='store', dest='model_path')
    parser.add_option("--test-file", action='store', dest='test_file')
    parser.add_option('--transcribe-file', action='store', dest='transcribe_file')
    parser.add_option('--output-file', action='store', dest='output_file')
    parser.add_option('--model', action='store', dest='model_base')

    (params, _) = parser.parse_args(sys.argv)
    if params.test_file and params.model_path:
        _eval(params)
    elif params.transcribe_file:
        _transcribe(params)
    else:
        _start_train(params)
