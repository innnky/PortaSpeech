
from string import punctuation
import re
from g2p_en import G2p

from text.symbols import word_boundary_symbol


def en_to_phonemes(text, add_word_boundary=False):
    text = text.rstrip(punctuation)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w in ["", " "]:
            continue
        if add_word_boundary:
            phones.append(word_boundary_symbol)
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))

    return phones


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

lexicon = read_lexicon("text/en_dict.dict")
