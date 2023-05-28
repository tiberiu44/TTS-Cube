import optparse
import pathlib
import sys
import tqdm

sys.path.append('')
from cube.networks.g2p import SimpleTokenizer
import json
import librosa
import soundfile as sf
from cube.io_utils.vocoder import MelVocoder
import numpy as np
from PIL import Image
import pysptk
import os
import textgrid
import numpy as np

tokenizer = SimpleTokenizer()


def _cost(t1, t2):
    t1 = t1['text'].lower()
    t2 = t2.word.lower()
    if t1 == t2:
        return 0
    if t1.startswith(t2) or t2.startswith(t1):
        return 0.5
    if t1.endswith(t2) or t2.endswith(t1):
        return 0.5

    return 1


def _align(tg_words, tok_words):
    a = np.zeros((len(tg_words) + 1, len(tok_words) + 1))
    for ii in range(a.shape[0]):
        a[ii, 0] = ii
    for ii in range(a.shape[1]):
        a[0, ii] = ii

    for ii in range(1, a.shape[0]):
        for jj in range(1, a.shape[1]):
            cost = _cost(tg_words[ii - 1], tok_words[jj - 1])
            mm = min([a[ii - 1, jj - 1], a[ii - 1, jj], a[ii, jj - 1]])
            a[ii, jj] = mm + cost

    ii = a.shape[0] - 1
    jj = a.shape[1] - 1
    tg2tok = [0 for _ in range(len(tg_words))]
    tg2tok[ii - 1] = jj - 1
    while ii > 1 or jj > 1:
        if ii == 1:
            jj -= 1
        elif jj == 1:
            ii -= 1
        elif a[ii - 1, jj - 1] <= a[ii - 1, jj] and a[ii - 1, jj - 1] <= a[ii, jj - 1]:
            ii -= 1
            jj -= 1
        elif a[ii - 1, jj] <= a[ii, jj - 1]:
            ii -= 1
        else:
            jj -= 1
        print(ii, jj)
        tg2tok[ii - 1] = jj - 1

    return tg2tok


def _merge(aligned_words, aligned_phons, tokenized_words):
    hybrid = []
    phon2word = []
    frame2phon = []

    tg2tok = _align(aligned_words, tokenized_words)

    tok2tg = {}
    for ii in range(len(tg2tok)):
        tok2tg[tg2tok[ii]] = ii

    linear = []
    for ii in range(len(tokenized_words)):
        word = tokenized_words[ii].word
        if ii not in tok2tg:
            obj = {
                'word': word,
                'phones': [{'phon': word, 'dur': 0}]
            }
        else:
            phonemes = []
            w_start = aligned_words[tok2tg[ii]]['start']
            w_end = aligned_words[tok2tg[ii]]['stop']
            for phone in aligned_phons:
                if phone['start'] >= w_start and phone['stop'] <= w_end:
                    phonemes.append({
                        'phon': phone['text'],
                        'dur': phone['stop'] - phone['start']
                    })
            obj = {
                'word': word,
                'phones': phonemes
            }
        linear.append(obj)

    for w in linear:
        print(w['word'], ' '.join([str(p['phon']) for p in w['phones']]))
    from ipdb import set_trace
    set_trace()
    return hybrid, phon2word, frame2phon


def render_spectrogram(mgc, output_file):
    bitmap = np.zeros((mgc.shape[1], mgc.shape[0], 3), dtype=np.uint8)
    mgc_min = mgc.min()
    mgc_max = mgc.max()

    for x in range(mgc.shape[0]):
        for y in range(mgc.shape[1]):
            val = (mgc[x, y] - mgc_min) / (mgc_max - mgc_min)

            color = val * 255
            bitmap[mgc.shape[1] - y - 1, x] = [color, color, color]

    img = Image.fromarray(bitmap)  # smp.toimage(bitmap)
    img.save(output_file)


def _import_audio(dataset, output_folder, input_folder, sample_rate, hop_size, prefix):
    vocoder = MelVocoder()
    wav = None
    last_file = None
    dataset.sort(key=lambda x: x['orig_filename'])
    oms = sample_rate / 1000
    for ii in tqdm.tqdm(range(len(dataset)), ncols=80):
        item = dataset[ii]
        id = "{0}_{1:08d}".format(prefix, ii)
        item['id'] = id
        if last_file != item['orig_filename']:
            wav, _ = librosa.load('{0}/{1}.wav'.format(input_folder, item['orig_filename']), sr=sample_rate)
            last_file = item['orig_filename']
        audio_segment = wav[int(item['orig_start'] * oms):int(item['orig_end'] * oms)]
        audio_segment = (audio_segment / (np.max(np.abs(audio_segment)))) * 0.98
        mel = vocoder.melspectrogram(audio_segment, sample_rate, 80, hop_size, False)
        output_base = '{0}/{1}'.format(output_folder, id)
        render_spectrogram(mel, '{0}.png'.format(output_base))
        sf.write('{0}.wav'.format(output_base), np.asarray(audio_segment * 32767, dtype=np.int16), sample_rate)
        np.save(open('{0}.mgc'.format(output_base), 'wb'), mel)
        json.dump(item, open('{0}.json'.format(output_base), 'w'))
        pitch = pysptk.rapt(audio_segment * 32767, sample_rate, hopsize=hop_size, min=60, max=400)
        np.save(open('{0}.pitch'.format(output_base), 'wb'), pitch)


def _get_all_files(folder):
    all_files = []
    for folder, subs, files in os.walk(folder):
        for filename in files:
            tgfile = os.path.join(folder, filename)
            if filename.lower().endswith('.textgrid'):
                wavfile = tgfile[:-9] + '.wav'
                if os.path.exists(wavfile):
                    all_files.append(tgfile[:-9])
    return all_files


def _import_dataset(params):
    dataset = []
    print("Search input folder for valid files")
    all_files = _get_all_files(params.input_folder)
    print(f"Found {len(all_files)} aligned files")
    for iFiles in tqdm.tqdm(range(len(all_files)), ncols=120):
        tg_file = all_files[iFiles] + '.TextGrid'
        tg = textgrid.TextGrid.fromFile(tg_file)
        wav_file = all_files[iFiles] + '.wav'
        orig_text = '§' + tg[2][0].mark
        norm_words = []
        for ii in range(len(tg[0])):
            w_tok = {
                'text': tg[0][ii].mark,
                'start': tg[0][ii].minTime,
                'stop': tg[0][ii].maxTime
            }
            norm_words.append(w_tok)
        phons = []
        for jj in range(len(tg[1])):
            p_tok = {
                'text': tg[1][jj].mark,
                'start': tg[1][jj].minTime,
                'stop': tg[1][jj].maxTime
            }
            phons.append(p_tok)

        tok_words = tokenizer(orig_text)
        hybrid, phon2word, frame2phone = _merge(norm_words, phons, tok_words)
        from ipdb import set_trace
        set_trace()
        item = {
            'orig_start': 0,
            'orig_end': 0,
            'orig_filename': all_files[iFiles],
            'orig_text': orig_text,
            'phones': hybrid,
            'words': tok_words,
            'phon2word': phon2word,
            'frame2phon': frame2phone,
            'speaker': params.speaker
        }
        dataset.append(item)
    # creating context
    for ii in range(len(dataset)):
        l_start = max(0, ii - params.prev_sentences)
        l_end = min(len(dataset), ii + params.next_sentences + 1)
        # shrink window if we are at the beginning or end of a chapter - context not relevant here
        for jj in range(l_start, ii):
            if dataset[ii]['orig_filename'] != dataset[jj]['orig_filename']:
                l_start += 1
        for jj in range(l_end, ii, 1):
            if dataset[ii]['orig_filename'] != dataset[jj]['orig_filename']:
                l_end -= 1
        left_context = ' '.join([item['orig_text'][1:] for item in dataset[l_start:ii]])
        right_context = ' '.join([item['orig_text'][1:] for item in dataset[ii + 1:l_end]])
        dataset[ii]['left_context'] = left_context
        dataset[ii]['right_context'] = right_context

    # train-dev split
    trainset = []
    devset = []
    split = int(1.0 / params.dev_ratio)
    if split == 0:
        print("Warning: Invalid value for dev-ratio. Everything will be in the training set.")
        trainset = dataset
    elif split == 1:
        print("Warning: Invalid value for dev-ratio. Everything will be in the devset set.")
        devset = dataset
    else:
        for ii in range(len(dataset)):
            if (ii + 1) % split == 0:
                devset.append(dataset[ii])
            else:
                trainset.append(dataset[ii])

    import datetime
    print("Found {0} valid sentences, with a total audio time of {1}.".format(valid_sents, datetime.timedelta(
        seconds=(total_time / 1000))))
    print("Trainset will contain {0} examples and devset {1} examples".format(len(trainset), len(devset)))
    input_folder = params.input_file[:params.input_file.rfind('/')]
    print("Processing trainset")
    _import_audio(trainset, "data/processed/train/", input_folder, params.sample_rate, params.hop_size, params.prefix)
    print("Processing devset")
    _import_audio(devset, "data/processed/dev/", input_folder, params.sample_rate, params.hop_size, params.prefix)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-folder', action='store', dest='input_folder',
                      help='File with alignments')
    # parser.add_option('--prev-sentences', type='int', dest='prev_sentences', default=5,
    #                   help='How many previous sentences to use for context (default=5)')
    # parser.add_option('--next-sentences', type='int', dest='next_sentences', default=5,
    #                   help='How of the following sentences to use for context (default=5)')
    parser.add_option('--dev-ratio', type='float', dest='dev_ratio', default=0.001,
                      help='Ratio between dev and train (default=0.001)')
    parser.add_option('--speaker', action='store', dest='speaker', default="none",
                      help='What label to use for the speaker (default="none")')
    parser.add_option('--sample-rate', type='int', dest='sample_rate', default=24000,
                      help='Upsample or downsample data to this sample-rate (default=24000)')
    parser.add_option('--hop-size', type='int', dest='hop_size', default=240,
                      help='Frame analysis hop-size (default=240)')
    parser.add_option('--prefix', dest='prefix', default='FILE',
                      help='What prefix to use for the filenames')

    (params, _) = parser.parse_args(sys.argv)
    if params.input_folder:
        _import_dataset(params)
    else:
        parser.print_help()
