import os.path
import random

import torch
import torch.nn as nn
import numpy as np
import sys
import json
from os import listdir
from os.path import isfile, join, exists
import librosa

import tqdm

sys.path.append('')

from torch.utils.data.dataset import Dataset
from cube.networks.g2p import SimpleTokenizer
from cube.utils.hf import HFTokenizer
import fasttext.util
import fasttext


class CubeganDataset(Dataset):
    def __init__(self, base_path: str, hf_model: str = None):
        self._base_path = base_path
        self._examples = []
        train_files_tmp = [join(base_path, f) for f in listdir(base_path) if isfile(join(base_path, f))]
        tok = SimpleTokenizer()
        if hf_model is not None:
            self._hf_tok = HFTokenizer(hf_model)
        else:
            self._hf_tok = None

        for file in tqdm.tqdm(train_files_tmp, desc='\tLoading dataset', ncols=80):
            if file[-4:] == '.mgc':
                bpath = file[:-4]
                # check all files exist
                json_file = '{0}.json'.format(bpath)
                pitch_file = '{0}.pitch'.format(bpath)
                if os.path.exists(json_file) and os.path.exists(pitch_file):
                    example = json.load(open(json_file))
                    # check if we have long alignments
                    durs = np.zeros((len(example['phones'])))
                    for index in example['frame2phon']:
                        durs[index] += 1
                    if max(durs) > 400:
                        continue
                    tmp = tok(example['left_context'])
                    example['words_left'] = [w.word for w in tmp]
                    tmp = tok(example['right_context'])
                    example['words_right'] = [w.word for w in tmp]
                    valid = True
                    if self._hf_tok is not None:
                        example['words_hf'] = self._hf_tok(example['words'])
                        example['words_left_hf'] = self._hf_tok(example['words_left'])
                        example['words_right_hf'] = self._hf_tok(example['words_right'])
                        if len(example['words_hf']['toks']) >= 512 or \
                                len(example['words_left_hf']['toks']) >= 512 or \
                                len(example['words_right_hf']['toks']) >= 512:
                            valid = False
                        for ii in example['words_hf']['word2tok']:
                            if example['words_hf']['word2tok'][ii] < 0 or \
                                    example['words_hf']['word2tok'][ii] >= len(example['words_hf']['toks']):
                                valid = False
                                break
                        for ii in example['words_left_hf']['word2tok']:
                            if example['words_left_hf']['word2tok'][ii] < 0 or \
                                    example['words_left_hf']['word2tok'][ii] >= len(example['words_left_hf']['toks']):
                                valid = False
                                break
                        for ii in example['words_right_hf']['word2tok']:
                            if example['words_right_hf']['word2tok'][ii] < 0 or \
                                    example['words_right_hf']['word2tok'][ii] >= len(example['words_right_hf']['toks']):
                                valid = False
                                break
                if valid:
                    self._examples.append(example)

    def __len__(self):
        return len(self._examples)

    def _make_absolute_silence(self, audio, pitch, meta):
        max_phone = max(meta['frame2phon'])
        frame2phon = meta['frame2phon']
        zero_frame = np.zeros((240))
        for iFrame in range(len(frame2phon)):
            if frame2phon[iFrame] == 0 or frame2phon[iFrame] == max_phone:
                start = iFrame * 240
                stop = start + 240
                audio[start:stop] = zero_frame
                pitch[iFrame] = 0
        return audio, pitch

    def __getitem__(self, item):
        description = self._examples[item]
        base_fn = '{0}/{1}'.format(self._base_path, description['id'])
        mgc = np.load('{0}.mgc'.format(base_fn))
        pitch = np.load('{0}.pitch'.format(base_fn))
        audio, sr = librosa.load('{0}.wav'.format(base_fn), sr=24000)
        audio, pitch = self._make_absolute_silence(audio, pitch, description)

        return {
            'meta': description,
            'mgc': mgc,
            'pitch': pitch,
            'audio': audio
        }


class CubeganEncodings:
    def __init__(self, filename: str = None):
        self.speaker2int = {}
        self.phon2int = {}
        self.max_duration = 0
        self.max_pitch = 0
        if filename is not None:
            self.load(filename)

    def compute(self, dataset: CubeganDataset):
        for example in tqdm.tqdm(dataset, ncols=80, desc='Computing encodings'):
            speaker = example['meta']['speaker']
            if speaker not in self.speaker2int:
                self.speaker2int[speaker] = len(self.speaker2int)
            for phone in example['meta']['phones']:
                if phone not in self.phon2int:
                    self.phon2int[phone] = len(self.phon2int)
            m_pitch = np.max(example['pitch'])
            if m_pitch > self.max_pitch:
                self.max_pitch = m_pitch
            durs = np.zeros((len(example['meta']['phones'])), dtype=np.long)
            for item in example['meta']['frame2phon']:
                durs[item] += 1
            m_dur = np.max(durs)
            if m_dur > self.max_duration:
                self.max_duration = m_dur

    def load(self, filename: str):
        input_obj = json.load(open(filename))
        self.speaker2int = input_obj['speaker2int']
        self.phon2int = input_obj['phon2int']
        self.max_pitch = input_obj['max_pitch']
        self.max_duration = input_obj['max_duration']

    def save(self, filename: str):
        output_obj = {
            'speaker2int': self.speaker2int,
            'phon2int': self.phon2int,
            'max_duration': int(self.max_duration),
            'max_pitch': int(self.max_pitch)
        }
        json.dump(output_obj, open(filename, 'w'))


class CubeganCollate:
    def __init__(self, encodings: CubeganEncodings, conditioning_type=None, training=True):
        self._encodings = encodings
        self._ignore_index = int(max(encodings.max_pitch, encodings.max_duration) + 1)
        self._conditioning_type = None
        self._training = training
        if conditioning_type is not None and conditioning_type.startswith('fasttext'):
            lang = conditioning_type.split(':')[-1]
            fasttext.util.download_model(lang, if_exists='ignore')
            self._ft = fasttext.load_model('cc.{0}.300.bin'.format(lang))
            self._conditioning_type = 'fasttext'
        elif conditioning_type is not None and conditioning_type.startswith('hf'):
            self._conditioning_type = 'hf'

    def collate_fn(self, batch):
        max_char = max([len(example['meta']['phones']) for example in batch])
        max_mel = max([example['mgc'].shape[0] for example in batch])
        x_char = np.zeros((len(batch), max_char))
        x_words = None
        tok_ids = None
        word2tok = None
        if self._conditioning_type == 'fasttext':
            x_words = self._get_ft_embeddings(batch)
        elif self._conditioning_type == 'hf':
            tok_ids, word2tok = self._get_hf_ids(batch)
        x_phoneme2word = np.zeros((len(batch), max_char), dtype='long')
        y_mgc = np.ones((len(batch), max_mel, 80)) * -5
        x_speaker = np.zeros((len(batch), 1))
        y_dur = np.zeros((len(batch), max_char))
        y_pitch = np.zeros((len(batch), max_mel))
        y_frame2phone = []  # Hop-size
        y_audio = np.zeros((len(batch), max_mel * 240), dtype='float')
        for ii in range(len(batch)):
            example = batch[ii]
            y_mgc[ii, :example['mgc'].shape[0], :] = example['mgc']
            x_speaker[ii] = self._encodings.speaker2int[example['meta']['speaker']] + 1
            for jj in range(len(example['meta']['phones'])):
                phoneme = example['meta']['phones'][jj]
                if phoneme in self._encodings.phon2int:
                    x_char[ii, jj] = self._encodings.phon2int[phoneme] + 1
            y_frame2phone.append(example['meta']['frame2phon'])
            phone2word = example['meta']['phon2word']
            # this works for fasttext, not sure about bert
            if self._conditioning_type == 'fasttext':
                x_phoneme2word[ii, :len(phone2word)] = np.array(phone2word) + len(example['meta']['words_left'])
            else:
                x_phoneme2word[ii, :len(phone2word)] = np.array(phone2word)  # the RNN will only see bert aligned data
            for phone_idx in y_frame2phone[-1]:
                y_dur[ii, phone_idx] += 1
            for jj in range(max_char - len(example['meta']['phones'])):
                y_dur[ii, len(example['meta']['phones']) + jj] = self._ignore_index
            y_pitch[ii, :example['pitch'].shape[0]] = example['pitch']
            if 'audio' in example:
                audio = example['audio']
                m_size = min(y_audio.shape[1], audio.shape[0])
                y_audio[ii, :m_size] = audio[:m_size]

        if x_words is not None:
            x_words = torch.tensor(x_words, dtype=torch.float)
        if tok_ids is not None:
            tok_ids = torch.tensor(tok_ids, dtype=torch.long)
        for iExample in range(y_dur.shape[0]):
            m_size = len(batch[iExample]['meta']['phones'])
            y_dur[iExample, :m_size] = np.clip(y_dur[iExample, :m_size], 0, 100)  # clip durations to 1 second
        return {
            'x_char': torch.tensor(x_char, dtype=torch.long),
            'x_words': x_words,
            'x_tok_ids': tok_ids,
            'x_word2tok': word2tok,
            'x_phon2word': torch.tensor(x_phoneme2word),
            'x_speaker': torch.tensor(x_speaker, dtype=torch.long),
            'y_mgc': torch.tensor(y_mgc, dtype=torch.float),
            'y_frame2phone': y_frame2phone,
            'y_pitch': torch.tensor(y_pitch, dtype=torch.long),
            'y_dur': torch.tensor(y_dur, dtype=torch.long),
            'y_audio': torch.tensor(y_audio, dtype=torch.float)
        }

    def _get_ft_embeddings(self, batch):
        max_words = max([len(example['meta']['words']) +
                         len(example['meta']['words_left']) +
                         len(example['meta']['words_right']) for example in batch])
        x_words = np.zeros((len(batch), max_words, 300))
        for ii in range(len(batch)):
            all_words = batch[ii]['meta']['words_left'] + \
                        batch[ii]['meta']['words'] + \
                        batch[ii]['meta']['words_right']
            for jj in range(len(all_words)):
                x_words[ii, jj, :] = self._ft.get_word_vector(str(all_words[jj]))
        return x_words

    def _get_hf_ids(self, batch):
        max_toks = max([len(example['meta']['words_hf']['tok_ids']) +
                        len(example['meta']['words_left_hf']['tok_ids']) +
                        len(example['meta']['words_right_hf']['tok_ids']) for example in batch])

        # we use coordinates for runtime, where bert will be run on word windows
        toks = np.zeros((len(batch), min(512, max_toks)), dtype='long')
        word2tok = []
        for ii in range(len(batch)):
            e_w2k = {}
            example = batch[ii]
            l_toks = example['meta']['words_left_hf']['tok_ids']
            l_size = len(l_toks)
            c_toks = example['meta']['words_hf']['tok_ids']
            c_size = len(c_toks)
            r_toks = example['meta']['words_right_hf']['tok_ids']
            r_size = len(r_toks)
            if l_size + c_size <= 512:
                start = 0
                offset = l_size
            else:
                start = c_size + l_size - 512
                offset = l_size - start
            # offset = l_size - start
            e_toks = l_toks + c_toks + r_toks
            e_toks = e_toks[start:]
            toks[ii, :min(toks.shape[1], len(e_toks))] = e_toks[:min(toks.shape[1], len(e_toks))]
            for word in example['meta']['words_hf']['word2tok']:
                e_w2k[word] = (ii, example['meta']['words_hf']['word2tok'][word] + offset)
            word2tok.append(e_w2k)

        return toks, word2tok
