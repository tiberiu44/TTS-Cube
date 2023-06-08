import sys
import torch

sys.path.append('')

from cube.io_utils.io_phonemizer import PhonemizerEncodings, PhonemizerCollate
from cube.networks.phonemizer import CubenetPhonemizerM2M, CubenetPhonemizer
from cube.networks.g2p import SimpleTokenizer


class Text2FeatBlizzard:
    def __init__(self, phonemizer_path: str):
        self._encodings = PhonemizerEncodings('{0}.encodings'.format(phonemizer_path))
        self._phonemizer = CubenetPhonemizer(self._encodings)
        self._phonemizer.load('{0}.model'.format(phonemizer_path))
        self._phonemizer.eval()
        self._tokenizer = SimpleTokenizer()
        self._collate = PhonemizerCollate(self._encodings)
        self._grapheme_list = [' '] * len(self._encodings.phonemes)

        for g in self._encodings.phonemes:
            self._grapheme_list[self._encodings.phonemes[g]] = g

    def __call__(self, text):
        text = text.replace('\n\n', '§')
        text = text.replace('\n', ' ')

        if not text.startswith('§'):
            text = '§' + text
        if not text[-1] == '§':
            text = text + '§'

        words = self._tokenizer(text)

        with torch.no_grad():
            X = self._collate.collate_fn([{'orig_text': text, 'phones': ['1']}])
            y_pred = torch.argmax(self._phonemizer(X), dim=-1)
            phonemes = [self._grapheme_list[index] for index in y_pred.squeeze().detach().numpy()]

        phon2word = []
        w_index = 0
        c_pos = 0
        currated_phonemes = []
        words = [w.word for w in words]
        for ii in range(len(phonemes)):

            if phonemes[ii] != '_':
                currated_phonemes.append(phonemes[ii])
                phon2word.append(w_index)
            c_pos += 1
            if c_pos == len(words[w_index]):
                c_pos = 0
                w_index += 1
        return {
            'orig_text': text,
            'words': words,
            'phones': currated_phonemes,
            'phon2word': phon2word
        }


class Text2Feat:
    def __init__(self, phonemizer_path: str):
        self._encodings = PhonemizerEncodings('{0}.encodings'.format(phonemizer_path))
        self._phonemizer = CubenetPhonemizerM2M(self._encodings)
        self._phonemizer.load('{0}.model'.format(phonemizer_path))
        self._phonemizer.eval()
        self._tokenizer = SimpleTokenizer()
        self._collate = PhonemizerCollate(self._encodings)
        self._grapheme_list = [' '] * len(self._encodings.phonemes)

        for g in self._encodings.phonemes:
            self._grapheme_list[self._encodings.phonemes[g]] = g

    def __call__(self, text):
        text = text.replace('\n\n', ' ')
        text = text.replace('\n', ' ')

        if not text.startswith(' '):
            text = ' ' + text
        if not text[-1] == ' ':
            text = text + ' '

        words = self._tokenizer(text)

        with torch.no_grad():
            X = self._collate.collate_fn(
                [{'orig_text': text, 'phones': ['1', '2', '3'], 'phon2word': [0, 0, 0],
                  'words': [w.word for w in words]}])
            del X['y_phon']
            y_phon_pred, y_nw_pred = self._phonemizer(X)  # torch.argmax(, dim=-1)
            y_phon_pred = torch.argmax(y_phon_pred, dim=-1)
            y_nw_pred = torch.argmax(y_nw_pred, dim=-1)
            phonemes = [self._grapheme_list[index] for index in y_phon_pred.squeeze().detach().numpy()]
        from ipdb import set_trace
        set_trace()
        phon2word = []
        w_index = 0
        c_pos = 0
        currated_phonemes = []
        words = [w.word for w in words]
        for ii in range(len(phonemes)):

            if phonemes[ii] != '_':
                currated_phonemes.append(phonemes[ii])
                phon2word.append(w_index)
            c_pos += 1
            if c_pos == len(words[w_index]):
                c_pos = 0
                w_index += 1
        return {
            'orig_text': text,
            'words': words,
            'phones': currated_phonemes,
            'phon2word': phon2word
        }


if __name__ == '__main__':
    text2feat = Text2Feat('data/phonemizer-en')
    rez = text2feat('This is a simple test.')
    for ii in range(len(rez['phones'])):
        phon = rez['phones'][ii]
        word = rez['words'][rez['phon2word'][ii]]
        print(phon, "\t", word)
