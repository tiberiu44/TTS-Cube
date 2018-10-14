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


def create_lab_input(txt_file, speaker_ident):
    seq = []
    fin = open(txt_file, 'r')
    line = fin.readline().strip().replace('\t', ' ')
    while True:
        nl = line.replace('  ', ' ')
        if nl == line:
            break
        line = nl

    # fout.write('START\n')
    from io_modules.dataset import PhoneInfo

    seq.append(PhoneInfo('START', [], 0, 0))
    # sys.stdout.write('START\n')
    for char in line:
        l_char = char.lower()
        style = 'CASE:lower'
        if l_char == l_char.upper():
            style = 'CASE:symb'
        elif l_char != char:
            style = 'CASE:upper'
        speaker = 'SPEAKER:' + speaker_ident
        seq.append(PhoneInfo(l_char, [speaker, style], 0, 0))
        # sys.stdout.write(l_char + '\t' + speaker + '\t' + style + '\n')

    seq.append(PhoneInfo('STOP', [], 0, 0))
    # sys.stdout.write('STOP\n')

    fin.close()
    return seq


def _render_spectrogram(mgc, output_file):
    bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
    mgc_min = mgc.min()
    mgc_max = mgc.max()

    for x in range(mgc.shape[0]):
        for y in range(mgc.shape[1]):
            val = (mgc[x, y] - mgc_min) / (mgc_max - mgc_min)

            color = val * 255
            bitmap[mgc.shape[1] - y - 1, x] = [color, color, color]
    import scipy.misc as smp

    img = smp.toimage(bitmap)
    img.save(output_file)


def synthesize(speaker, input_file, output_file, params):
    print ("[Encoding]")
    from io_modules.dataset import Dataset
    from io_modules.dataset import Encodings
    from models.encoder import Encoder
    from trainers.encoder import Trainer
    encodings = Encodings()
    encodings.load('data/models/encoder.encodings')
    encoder = Encoder(params, encodings, runtime=True)
    encoder.load('data/models/rnn_encoder')

    seq = create_lab_input(input_file, speaker)
    mgc, att = encoder.generate(seq)
    _render_spectrogram(mgc, output_file + '.png')

    print ("[Vocoding]")
    from models.vocoder import Vocoder
    from trainers.vocoder import Trainer
    vocoder = Vocoder(params, runtime=True)
    vocoder.load('data/models/rnn_vocoder')

    import time
    start = time.time()
    signal = vocoder.synthesize(mgc, batch_size=1000, temperature=params.temperature, sample=params.sample)
    stop = time.time()
    sys.stdout.write(" execution time=" + str(stop - start))
    sys.stdout.write('\n')
    sys.stdout.flush()
    from io_modules.dataset import DatasetIO
    dio = DatasetIO()
    enc = dio.b16_dec(signal, discreete=True)
    dio.write_wave(output_file, enc, params.target_sample_rate)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-file', action='store', dest='txt_file',
                      help='Path to the text file that will be synthesized')
    parser.add_option('--speaker', action='store', dest='speaker',
                      help='Speaker identity')
    parser.add_option('--output-file', action='store', dest='output_file',
                      help='Output WAVE file')
    parser.add_option("--batch-size", action='store', dest='batch_size', default='1000', type='int',
                      help='number of samples in a single batch (default=1000)')
    parser.add_option("--set-mem", action='store', dest='memory', default='2048', type='int',
                      help='preallocate memory for batch training (default 2048)')
    parser.add_option("--use-gpu", action='store_true', dest='gpu',
                      help='turn on/off GPU support')
    parser.add_option("--sample", action='store_true', dest='sample',
                      help='Use random sampling')
    parser.add_option('--mgc-order', action='store', dest='mgc_order', type='int',
                      help='Order of MGC parameters (default=60)', default=60)
    parser.add_option('--temperature', action='store', dest='temperature', type='float',
                      help='Exploration parameter (max 1.0, default 0.8)', default=0.8)
    parser.add_option('--target-sample-rate', action='store', dest='target_sample_rate',
                      help='Resample input files at this rate (default=16000)', type='int', default=16000)

    (params, _) = parser.parse_args(sys.argv)

    if not params.speaker:
        print ("Speaker identity is mandatory")
    elif not params.txt_file:
        print ("Input file is mandatory")
    elif not params.output_file:
        print ("Output file is mandatory")

    memory = int(params.memory)
    # for compatibility we have to add this paramater
    params.learning_rate = 0.0001
    dynet_config.set(mem=memory, random_seed=9)
    if params.gpu:
        dynet_config.set_gpu()

    synthesize(params.speaker, params.txt_file, params.output_file, params)
