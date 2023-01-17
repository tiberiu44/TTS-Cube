import optparse
import sys

sys.path.append('')
from cube.networks.g2p import SimpleTokenizer

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
    import tqdm
    lines = open(params.input_file).readlines()
    valid_sents = 0
    total_time = 0
    dataset = []
    for ii in tqdm.tqdm(range(len(lines))):
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
        total_time += sum(durs)
        valid_sents += 1
        # for ii in range(len(hybrid)):
        #     text = hybrid[ii] + " " + words[phon2word[ii]] + " -"
        #     for jj in range(len(frame2phone)):
        #         if frame2phone[jj] == ii:
        #             text += " " + str(jj)
        #     print(text)
        item = {
            'orig_start': int(parts[1]),
            'orig_end': int(parts[2]),
            'orig_filename': parts[0],
            'orig_text': text,
            'phones': hybrid,
            'words': words,
            'phon2word': phon2word,
            'frame2phon': frame2phone
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
    from ipdb import set_trace
    set_trace()

    import datetime
    print("Found {0} valid sentences, with a total audio time of {1}.".format(valid_sents, datetime.timedelta(
        seconds=(total_time / 1000))))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-file', action='store', dest='input_file',
                      help='File with alignments')
    parser.add_option('--prev-sentences', type='int', dest='prev_sentences', default=5,
                      help='How many previous sentences to use for context (default=5)')
    parser.add_option('--next-sentences', type='int', dest='next_sentences', default=5,
                      help='How of the following sentences to use for context (default=5)')
    parser.add_option('--dev-ratio', type='float', dest='dev_ratio', default=0.001,
                      help='Ratio between dev and train (default=0.001)')
    parser.add_option('--speaker', action='store', dest='speaker', default="none",
                      help='What label to use for the speaker (default="none")')

    (params, _) = parser.parse_args(sys.argv)
    if params.input_file:
        _import_dataset(params)
    else:
        parser.print_help()
