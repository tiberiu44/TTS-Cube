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
import numpy as np

sys.path.append('')
from cube.networks.g2p import G2P


def _normalize(data):
    m = np.max(np.abs(data))
    data = (data / m) * 0.999
    return data


def array2file(a, filename):
    np.save(filename, a)


def file2array(filename):
    a = np.load(filename)
    return a


def render_spectrogram(mgc, output_file):
    bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
    mgc_min = mgc.min()
    mgc_max = mgc.max()

    for x in range(mgc.shape[0]):
        for y in range(mgc.shape[1]):
            val = (mgc[x, y] - mgc_min) / (mgc_max - mgc_min)

            color = val * 255
            bitmap[mgc.shape[1] - y - 1, x] = [color, color, color]
    from PIL import Image

    img = Image.fromarray(bitmap)  # smp.toimage(bitmap)
    img.save(output_file)


def _is_match(phones, trans):
    for ii in range(len(trans)):
        if phones[ii] != trans[ii]:
            return False
    return True


def _align(phs_data, transcription, mgc):
    s1 = transcription
    s2 = [p.split(' ')[2].strip() for p in phs_data]
    start = [int(p.split(' ')[0].strip()) for p in phs_data]
    stop = [int(p.split(' ')[1].strip()) for p in phs_data]
    a = np.zeros((len(s1) + 1, len(s2) + 1))
    for ii in range(a.shape[0]):
        a[ii, 0] = ii
    for ii in range(a.shape[1]):
        a[0, ii] = ii

    for ii in range(1, a.shape[0]):
        for jj in range(1, a.shape[1]):
            cost = 0
            c_ph = s1[ii - 1]
            c_htk = s2[jj - 1]
            e_ph = _encode_htk(c_ph)

            if c_ph != c_htk and e_ph != c_htk:
                cost = 1
            # if s1[ii - 1] != s2[jj - 1] and s2[jj - 1][0] != '\\':
            #     cost = 1
            a[ii, jj] = cost + min([a[ii - 1, jj], a[ii - 1, jj - 1], a[ii, jj - 1]])

    ii = a.shape[0] - 1
    jj = a.shape[1] - 1
    phs2t = {jj - 1: ii - 1}
    while ii != 1 or jj != 1:
        if ii == 1:
            jj -= 1
        elif jj == 1:
            ii -= 1
        else:
            if a[ii - 1, jj - 1] <= a[ii - 1, jj] and a[ii - 1, jj - 1] <= a[ii, jj - 1]:
                ii -= 1
                jj -= 1
            elif a[ii - 1, jj] < a[ii - 1][jj - 1] and a[ii - 1, jj] < a[ii, jj - 1]:
                ii -= 1
            else:
                jj -= 1
        phs2t[jj - 1] = ii - 1
    trans2interval = {}
    start_i = 0
    for iPhs in range(len(phs_data)):
        if iPhs in phs2t:
            trans2interval[phs2t[iPhs]] = (start_i, int(stop[iPhs]))
            start_i = int(stop[iPhs])
    align = np.zeros(mgc.shape[0], dtype=np.long)
    align -= 1
    tpos = 0
    # fixing
    start = 0
    for tpos in range(len(transcription)):
        if tpos in trans2interval:
            trans2interval[tpos] = (start, trans2interval[tpos][1])
            start = trans2interval[tpos][1]
    for mIndex in range(align.shape[0]):
        t = mIndex * 16
        for tpos in trans2interval:
            if (t >= trans2interval[tpos][0] / 10000) and (t <= trans2interval[tpos][1] / 10000):
                align[mIndex] = tpos
                break
        if align[mIndex] == -1:
            align[mIndex] = len(transcription) - 1

    return align


def _encode_htk(string):
    """
    char *ReWriteString(char *s,char *dst, char q)
{
   static char stat[MAXSTRLEN*4];
   Boolean noSing,noDbl;
   int n;
   unsigned char *p,*d;

   if (dst==NULL) d=(unsigned char*)(dst=stat);
   else d=(unsigned char*)dst;

   if (s[0]!=SING_QUOTE) noSing=TRUE;
   else noSing=FALSE;
   if (s[0]!=DBL_QUOTE) noDbl=TRUE;
   else noDbl=FALSE;

   if (q!=ESCAPE_CHAR && q!=SING_QUOTE && q!=DBL_QUOTE) {
      q=0;
      if (!noDbl || !noSing) {
         if (noSing && !noDbl) q=SING_QUOTE;
         else q=DBL_QUOTE;
      }
   }
   if (q>0 && q!=ESCAPE_CHAR) *d++=q;
   for (p=(unsigned char*)s;*p;p++) {
      if (*p==ESCAPE_CHAR || *p==q ||
          (q==ESCAPE_CHAR && p==(unsigned char*)s &&
           (*p==SING_QUOTE || *p==DBL_QUOTE)))
         *d++=ESCAPE_CHAR,*d++=*p;
      else if (isprint(*p) || noNumEscapes) *d++=*p;
      else {
         n=*p;
         *d++=ESCAPE_CHAR;
         *d++=((n/64)%8)+'0';*d++=((n/8)%8)+'0';*d++=(n%8)+'0';
      }
   }
   if (q>0 && q!=ESCAPE_CHAR) *d++=q;
   *d=0;
   return(dst);
}
    :param str:
    :return:
    """
    s = ''
    bb = bytes(string, 'utf-8')
    for b in bb:
        s += '\{0}{1}{2}'.format((b // 64) % 8, (b // 8) % 8, b % 8)
    return s


def create_lab_file(txt_file, phs_file, mgc, lab_file, speaker_name=None, g2p=None, lang=None, emotion='None'):
    fin = open(txt_file, 'r', encoding='utf-8')
    line = fin.readline().strip().replace('\t', ' ')
    json_obj = {}
    while True:
        nl = line.replace('  ', ' ')
        if nl == line:
            break
        line = nl
    line = line.strip()

    if speaker_name is not None:
        json_obj['speaker'] = speaker_name  # speaker = 'SPEAKER:' + speaker_name
    elif len(txt_file.replace('\\', '/').split('/')[-1].split('_')) != 1:
        json_obj['speaker'] = txt_file.replace('\\', '/').split('_')[0].split('/')[-1]
    else:
        json_obj['speaker'] = 'none'

    json_obj['emotion'] = emotion
    json_obj['text'] = line
    if g2p is not None:
        trans = ['<START>']
        utt = g2p.transcribe_utterance(line)
        for word in utt:
            for ph in word.transcription:
                trans.append(ph)
        trans.append('<STOP>')
        json_obj['transcription'] = trans
    else:
        json_obj['transcription'] = ['<START>'] + [c.lower() for c in line] + ['<STOP>']

    phs_data = open(phs_file).readlines()
    fin.close()
    tmp = _align(phs_data, json_obj['transcription'], mgc)
    if tmp is None:
        return False
    json_obj['aligned'] = tmp.tolist()
    json_obj['lang'] = lang

    fout = open(lab_file, 'w', encoding='utf-8')
    import json
    json.dump(json_obj, fout)
    fout.close()
    return True


def _highpass_filter(y, sr):
    from scipy import signal
    sos = signal.butter(30, 100, 'hp', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, y)
    return filtered


def phase_1_prepare_corpus(params):
    from os import listdir
    from os.path import isfile, join
    from os.path import exists
    from pysptk import rapt

    train_files_tmp = [f for f in listdir(params.train_folder) if isfile(join(params.train_folder, f))]
    if params.dev_folder is not None:
        dev_files_tmp = [f for f in listdir(params.dev_folder) if isfile(join(params.dev_folder, f))]
    else:
        dev_files_tmp = []

    g2p = None
    if params.g2p:
        g2p = G2P()
        g2p.load(params.g2p)
        g2p.to(params.device)
        g2p.seq2seq.eval()

    sys.stdout.write("Scanning training files...")
    sys.stdout.flush()
    final_list = []
    for file in train_files_tmp:
        base_name = file[:-4]
        lab_name = base_name + '.txt'
        wav_name = base_name + '.wav'
        phs_name = base_name + '.phs'
        if exists(join(params.train_folder, lab_name)) and exists(join(params.train_folder, wav_name)) and exists(
                join(params.train_folder, phs_name)):
            if base_name not in final_list:
                final_list.append(base_name)

    train_files = final_list
    sys.stdout.write(" found " + str(len(train_files)) + " valid training files\n")
    sys.stdout.write("Scanning development files...")
    sys.stdout.flush()
    final_list = []
    for file in dev_files_tmp:
        base_name = file[:-4]
        lab_name = base_name + '.txt'
        wav_name = base_name + '.wav'
        if exists(join(params.dev_folder, lab_name)) and exists(join(params.dev_folder, wav_name)):
            if base_name not in final_list:
                final_list.append(base_name)

    dev_files = final_list
    sys.stdout.write(" found " + str(len(dev_files)) + " valid development files\n")
    from cube.io_utils.dataset import DatasetIO
    from cube.io_utils.vocoder import MelVocoder
    from shutil import copyfile
    dio = DatasetIO()

    vocoder = MelVocoder()
    base_folder = params.train_folder
    total_files = 0
    for index in range(len(train_files)):
        total_files += 1
        sys.stdout.write("\r\tprocessing file " + str(index + 1) + "/" + str(len(train_files)))
        sys.stdout.flush()
        base_name = train_files[index]
        txt_name = base_name + '.txt'
        wav_name = base_name + '.wav'
        spc_name = base_name + '.png'
        lab_name = base_name + '.lab'
        phs_name = base_name + '.phs'

        tgt_tt_name = txt_name
        tgt_spc_name = spc_name
        tgt_lab_name = lab_name
        if params.prefix is not None:
            tgt_txt_name = params.prefix + "_{:05d}".format(total_files) + '.txt'
            tgt_spc_name = params.prefix + "_{:05d}".format(total_files) + '.png'
            tgt_lab_name = params.prefix + "_{:05d}".format(total_files) + '.lab'

        # LAB - copy or create
        # TXT
        copyfile(join(base_folder, txt_name), join('data/processed/train', tgt_txt_name))
        # WAVE
        data, sample_rate = dio.read_wave(join(base_folder, wav_name), sample_rate=params.target_sample_rate)
        f0 = rapt(np.array(data * 32000, dtype=np.float32), params.target_sample_rate, 256, min=30, max=500)

        data = _normalize(data)
        data = _highpass_filter(data, params.target_sample_rate)

        mgc = vocoder.melspectrogram(data, sample_rate=params.target_sample_rate, num_mels=params.mgc_order)
        if not create_lab_file(join(base_folder, txt_name), join(base_folder, phs_name), mgc,
                               join('data/processed/train', tgt_lab_name), speaker_name=params.speaker, g2p=g2p,
                               lang=params.lang, emotion=params.emotion):
            continue
        # fft = vocoder.fft(data, sample_rate=params.target_sample_rate)
        # SPECT
        # render_spectrogram(mgc, join('data/processed/train', tgt_spc_name))
        if params.prefix is None:
            # dio.write_wave(join('data/processed/train', base_name + '.orig.wav'), data, sample_rate)
            array2file(mgc, join('data/processed/train', base_name + '.mgc'))
            array2file(f0, join('data/processed/train', base_name + '.f0'))
            # array2file(fft, join('data/processed/train', base_name + '.fft'))
        else:
            tgt_wav_name = params.prefix + "_{:05d}".format(total_files) + '.orig.wav'
            tgt_mgc_name = params.prefix + "_{:05d}".format(total_files) + '.mgc'
            tgt_fft_name = params.prefix + "_{:05d}".format(total_files) + '.fft'
            # dio.write_wave(join('data/processed/train', tgt_wav_name), data, sample_rate)
            array2file(mgc, join('data/processed/train', tgt_mgc_name))
            array2file(f0, join('data/processed/train', tgt_mgc_name.replace('.mgc', '.f0')))
            # array2file(fft, join('data/processed/train', tgt_fft_name))

    sys.stdout.write('\n')
    base_folder = params.dev_folder
    for index in range(len(dev_files)):
        total_files += 1
        sys.stdout.write("\r\tprocessing file " + str(index + 1) + "/" + str(len(dev_files)))
        sys.stdout.flush()
        base_name = dev_files[index]
        txt_name = base_name + '.txt'
        wav_name = base_name + '.wav'
        spc_name = base_name + '.png'
        lab_name = base_name + '.lab'
        phs_name = base_name + '.phs'

        tgt_txt_name = txt_name
        tgt_spc_name = spc_name
        tgt_lab_name = lab_name
        if params.prefix is not None:
            tgt_txt_name = params.prefix + "_{:05d}".format(total_files) + '.txt'
            tgt_spc_name = params.prefix + "_{:05d}".format(total_files) + '.png'
            tgt_lab_name = params.prefix + "_{:05d}".format(total_files) + '.lab'

        # LAB - copy or create

        # TXT
        copyfile(join(base_folder, txt_name), join('data/processed/dev', tgt_txt_name))
        # WAVE
        data, sample_rate = dio.read_wave(join(base_folder, wav_name), sample_rate=params.target_sample_rate)
        f0 = rapt(np.array(data * 32000, dtype=np.float32), params.target_sample_rate, 256, min=30, max=500)
        data = _normalize(data)
        mgc = vocoder.melspectrogram(data, sample_rate=params.target_sample_rate, num_mels=params.mgc_order)
        # fft = vocoder.fft(data, sample_rate=params.target_sample_rate)
        # SPECT
        if not create_lab_file(join(base_folder, txt_name), join(base_folder, phs_name), mgc,
                               join('data/processed/dev', tgt_lab_name), speaker_name=params.speaker, g2p=g2p,
                               lang=params.lang, emotion=params.emotion):
            continue

        # render_spectrogram(mgc, join('data/processed/dev', tgt_spc_name))
        if params.prefix is None:
            # dio.write_wave(join('data/processed/dev', base_name + '.orig.wav'), data, sample_rate)
            array2file(mgc, join('data/processed/dev', base_name + '.mgc'))
            array2file(f0, join('data/processed/dev', base_name + '.f0'))
            # array2file(fft, join('data/processed/dev', base_name + '.fft'))
        else:
            tgt_wav_name = params.prefix + "_{:05d}".format(total_files) + '.orig.wav'
            tgt_mgc_name = params.prefix + "_{:05d}".format(total_files) + '.mgc'
            tgt_fft_name = params.prefix + "_{:05d}".format(total_files) + '.fft'
            # dio.write_wave(join('data/processed/dev', tgt_wav_name), data, sample_rate)
            array2file(mgc, join('data/processed/dev', tgt_mgc_name))
            array2file(f0, join('data/processed/dev', tgt_mgc_name.replace('.mgc', '.f0')))
            # array2file(fft, join('data/processed/dev', tgt_fft_name))

    sys.stdout.write('\n')


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--cleanup', action='store_true', dest='cleanup',
                      help='Cleanup temporary training files and start from fresh')
    parser.add_option('--train-folder', action='store', dest='train_folder',
                      help='Location of the training files')
    parser.add_option('--dev-folder', action='store', dest='dev_folder',
                      help='Location of the development files')
    parser.add_option('--target-sample-rate', action='store', dest='target_sample_rate',
                      help='Resample input files at this rate (default=16000)', type='int', default=16000)
    parser.add_option('--mgc-order', action='store', dest='mgc_order', type='int',
                      help='Order of MGC parameters (default=80)', default=80)
    parser.add_option('--speaker', action='store', dest='speaker', help='Import data under given speaker')
    parser.add_option('--g2p', action='store', dest='g2p', help='What G2P model to use')
    parser.add_option('--device', action='store', dest='device', help='Device to use for g2p', default='cpu')
    parser.add_option('--prefix', action='store', dest='prefix', help='Use this prefix when importing files')
    parser.add_option('--lang', action='store', dest='lang', help='Language for multilingual setting', default='none')
    parser.add_option('--emotion', action='store', dest='emotion', default='neutral',
                      choices=['neutral', 'angry', 'anxious', 'apologetic', 'assertive', 'concerned',
                               'disgust', 'encouraging', 'excited', 'happy', 'sad', 'fear', 'surprised', 'unk'])

    (params, _) = parser.parse_args(sys.argv)

    phase_1_prepare_corpus(params)
