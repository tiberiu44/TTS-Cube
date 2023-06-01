from transformers import AutoTokenizer


class HFTokenizer:
    def __init__(self, model_name):
        self._tok = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, sentence):
        if len(sentence) == 0:
            return {'tok_ids': [],
                    'toks': [],
                    'word2tok': {}}
        orig_sent = [w for w in sentence]
        for ii in range(len(sentence)):
            if sentence[ii] == ' ':
                sentence[ii] = '_'
        iid = self._tok(sentence, add_special_tokens=False, is_split_into_words=True)['input_ids']
        toks = self._tok.convert_ids_to_tokens(iid)
        word2tok = {}
        tok_ids = []
        index_tok = 0
        index_word = 0
        # hack
        for ii in range(1, len(toks)):
            if toks[ii - 1] == '▁':
                toks[ii] = '▁' + toks[ii]
        while index_tok < len(toks) or index_word < len(orig_sent):
            # find compatible tok
            while index_tok < len(toks) and (
                    toks[index_tok].replace('▁', '').replace('_', '') == '' or not toks[index_tok].startswith('▁')):
                if toks[index_tok].replace('▁', '').replace('_', '') != '':
                    tok_ids.append(iid[index_tok])
                index_tok += 1
            # find compatible word
            while index_word < len(orig_sent) and orig_sent[index_word] == ' ':
                index_word += 1
            if index_tok < len(iid):
                tok_ids.append(iid[index_tok])
            word2tok[index_word] = len(tok_ids) - 1
            index_tok += 1
            index_word += 1

        surviving_toks = self._tok.convert_ids_to_tokens(tok_ids)

        return {'toks': surviving_toks,
                'tok_ids': tok_ids,
                'word2tok': word2tok}
