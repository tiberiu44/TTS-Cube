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

tokenizer = SimpleTokenizer()


def _merge(text, phon, durs):
    hybrid = []
    phon2word = []
    frame2phon = []
    words = [w.word for w in tokenizer(text)]
    windex = 0
    cindex = 0
    dd = []
    for t, p, d in zip(text, phon, durs):
        if t.isalpha():
            hybrid.append(p)
        else:
            hybrid.append(t)
        phon2word.append(windex)
        cindex += 1
        if cindex == len(words[windex]):
            cindex = 0
            windex += 1
        dd.append(d)

    nh = []
    nd = []
    np2w = []
    delta = 0
    for t, h, d, p2w in zip(text, hybrid, durs, phon2word):
        if t.isalpha() and t != ' ' and h == '_':
            delta += 1
            continue
        nh.append(h)
        nd.append(d)
        np2w.append(p2w)

    # hard-coded frameshift of 10
    durs = nd
    cphon = 0
    total = sum(durs)
    lstart = 0
    pstart = []
    pend = []
    last = 0
    for d in durs:
        pstart.append(last)
        pend.append(last + d)
        last = pend[-1]

    for frame in range(total // 10):
        while (frame * 10) >= pend[cphon]:
            cphon += 1
            while durs[cphon] == 0:
                cphon += 1
        frame2phon.append(cphon)
    return nh, words, np2w, frame2phon


def _import_dataset(params):
    lines = open(params.input_file).readlines()
    valid_sents = 0
    total_time = 0
    dataset = []
    print("Reading and processing alignment file")
    for ii in tqdm.tqdm(range(len(lines)), ncols=120):
        line = lines[ii].strip()
        parts = line.split('|')
        if len(parts) < 6:
            continue
        text = parts[3]
        if '{' in text and '}' in text:
            continue
        durs = [int(x) for x in parts[5].strip().split(' ')]
        phon = parts[4].split(' ')
        if len(text) != len(phon) or len(text) != len(durs):
            from ipdb import set_trace
            set_trace()
        hybrid, words, phon2word, frame2phone = _merge(text, phon, durs)
        for ii in range(len(phon)):
            if not text[ii].isalpha() and (phon[ii] == '_' or phon[ii] == '__'):
                phon[ii] = text[ii]
        total_time += sum(durs)
        valid_sents += 1

        item = {
            'orig_text': text,
            'hybrid': hybrid,
            'phones': phon,
            'words': words,
            'phon2word': phon2word,
        }
        dataset.append(item)

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
    json.dump(trainset, open('{0}.train'.format(params.output_base), 'w'))
    json.dump(devset, open('{0}.dev'.format(params.output_base), 'w'))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-file', action='store', dest='input_file',
                      help='File with alignments')
    parser.add_option('--dev-ratio', type='float', dest='dev_ratio', default=0.001,
                      help='Ratio between dev and train (default=0.001)')
    parser.add_option('--output-base', action='store', dest='output_base')

    (params, _) = parser.parse_args(sys.argv)
    if params.input_file:
        _import_dataset(params)
    else:
        parser.print_help()
