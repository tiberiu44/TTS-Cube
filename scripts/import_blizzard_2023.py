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
    import datetime
    print("Found {0} valid sentences, with a total audio time of {1}.".format(valid_sents, datetime.timedelta(
        seconds=(total_time / 1000))))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-file', action='store', dest='input_file',
                      help='File with alignments')
    parser.add_option('--speaker', action='store', dest='speaker', default="none",
                      help='What label to use for the speaker (default="none")')

    (params, _) = parser.parse_args(sys.argv)
    if params.input_file:
        _import_dataset(params)
    else:
        parser.print_help()
