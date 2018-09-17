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

import optparse
import sys

import dynet_config


def train(params):
    from io_modules.encodings import Encodings
    from io_modules.dataset import LTSDataset

    sys.stdout.write('Loading datasets...\n')
    trainset = LTSDataset(params.train_file)
    devset = LTSDataset(params.dev_file)
    sys.stdout.write('Trainset has ' + str(len(trainset.entries)) + ' entries\n')
    sys.stdout.write('Devset has ' + str(len(devset.entries)) + ' entries\n')
    encodings = Encodings()
    encodings.update_encodings(trainset)
    sys.stdout.write('Found ' + str(len(encodings.char2int)) + ' characters\n')
    sys.stdout.write('Found ' + str(len(encodings.phoneme2int)) + ' phonemes\n')

    from trainers.g2p import G2PTrainer
    from models.g2p import G2P

    model = G2P(encodings)
    trainer = G2PTrainer()
    trainer.start_training(model, encodings, trainset, devset, params.model_base, params.batch_size, params.patience)


def test(params):
    from io_modules.encodings import Encodings
    from io_modules.dataset import LTSDataset

    encodings = Encodings()
    encodings.load(params.model + '.encodings')
    testset = LTSDataset(params.input)
    from models.g2p import G2P
    model = G2P(encodings)
    model.load(params.model + '-bestAcc.network')
    f = open(params.output, 'w')
    correct = 0
    last_proc = 0
    index = 0
    for entry in testset.entries:
        index += 1
        curr_proc = int(index * 100 / len(testset.entries))
        if curr_proc % 5 == 0:
            while last_proc < curr_proc:
                last_proc += 5
                sys.stdout.write(str(last_proc) + ' ')
                sys.stdout.flush()
        p_transcription = model.transcribe(entry.word)
        ok = False
        if p_transcription == entry.transcription:
            correct += 1
            ok = True
        f.write(entry.word + '\t')
        for phon in entry.transcription:
            f.write(phon + ' ')
        f.write('\t')
        for phon in p_transcription:
            f.write(phon + ' ')
        if not ok:
            f.write('***')
        f.write('\n')
    f.close()
    sys.stdout.write('done\n')
    sys.stdout.write('Word level accuracy is ' + str(float(correct) / len(testset.entries)) + '\n')


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--train', action='store_true', dest='train', help='Train a new model')
    parser.add_option('--test', action='store_true', dest='test', help='Test the pretrained model')
    parser.add_option('--input-file', action='store', dest='input', help='Input File')
    parser.add_option('--output-file', action='store', dest='output', help='Output file')
    parser.add_option('--model', action='store', dest='model', help='Pretrained model file')

    parser.add_option("--batch-size", action='store', dest='batch_size', default='100', type='int',
                      help='number of samples in a single batch (default=100)')
    parser.add_option("--set-mem", action='store', dest='memory', default='2048', type='int',
                      help='preallocate memory for batch training (default 2048)')
    parser.add_option("--autobatch", action='store_true', dest='autobatch',
                      help='turn on/off dynet autobatching')
    parser.add_option("--use-gpu", action='store_true', dest='gpu',
                      help='turn on/off GPU support')
    parser.add_option('--train-file', action='store', dest='train_file',
                      help='Path to training file')
    parser.add_option('--dev-file', action='store', dest='dev_file',
                      help='Path to development file')
    parser.add_option('--store', action='store', dest='model_base',
                      help='Location where to store the model')
    parser.add_option("--patience", action='store', dest='patience', default='20', type='int',
                      help='Early stopping condition')

    (params, _) = parser.parse_args(sys.argv)

    memory = int(params.memory)
    if params.autobatch:
        autobatch = True
    else:
        autobatch = False
    dynet_config.set(mem=memory, random_seed=9, autobatch=autobatch)
    if params.gpu:
        dynet_config.set_gpu()

    import dynet as dy

    if params.train:
        train(params)
    elif params.test:
        test(params)
    else:
        print ("Invalid parameters. Use '--help' for help")
