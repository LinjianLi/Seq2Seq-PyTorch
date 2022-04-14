import os
import logging
import re
import json
from sklearn.model_selection import train_test_split
from seq2seq.inputter.vocab import Vocab

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Since there are a lot of example sentences and we want to train something quickly,
# we’ll trim the data set to only relatively short and simple sentences.

MAX_LENGTH = 15


def normalize_str(s):
    # return re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return re.sub(r"[^a-zA-Z]+", r" ", s)


def normalizePair(p):
    p = [normalize_str(s) for s in p]
    p = [s.lower() for s in p]
    return p


accept_eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def pair_is_simple(p):
    # If pair is simple, it will be kept. return True
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[0].startswith(accept_eng_prefixes)
# ----------------------------------------------------------


def create_vocab(vocab_file_eng, vocab_file_fra):
    if not os.path.exists(vocab_file_eng) or not os.path.exists(vocab_file_fra):
        vocab_eng = Vocab("eng")
        vocab_fra = Vocab("fra")
        logger.info('Creating vocab.')
        with open("./data/eng-fra.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.split('\t')
                line = normalizePair(line)
                if not pair_is_simple(line):
                    continue
                vocab_eng.add_sentence(line[0], to_lower=True, remove_punc=True)
                vocab_fra.add_sentence(line[1], to_lower=True, remove_punc=True)
        logger.info(vocab_eng)
        logger.info(vocab_fra)
        logger.info('Storing vocab.')
        vocab_eng.to_json(vocab_file_eng)
        vocab_fra.to_json(vocab_file_fra)
        return vocab_eng, vocab_fra


def get_vocab():
    vocab_file_eng = "vocab_eng.json"
    vocab_file_fra = "vocab_fra.json"
    if not os.path.exists(vocab_file_eng) or not os.path.exists(vocab_file_fra):
        return create_vocab(vocab_file_eng, vocab_file_fra)
    else:
        logger.info('Loading vocab.')
        vocab_eng = Vocab.from_json(vocab_file_eng)
        vocab_fra = Vocab.from_json(vocab_file_fra)
        logger.info(vocab_eng)
        logger.info(vocab_fra)
        return vocab_eng, vocab_fra


def prepare_data(data_path, vocab_eng, vocab_fra):
    data_ids = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split('\t')
            line = normalizePair(line)
            if not pair_is_simple(line):
                continue
            tgt, inp = line # French to English
            data_ids.append({"input": vocab_fra.indexes_from_sentence(inp, add_eos=True),
                             "target": vocab_eng.indexes_from_sentence(tgt, add_eos=True)})
    return data_ids


def get_train_val_data(data_file="./data/eng-fra.txt", vocab_eng=None, vocab_fra=None):
    logger.info('Preparing data.')
    train_data_path = 'train_data.json'
    val_data_path = 'val_data.json'
    split_val_ratio = 0.2
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        data = prepare_data(data_file, vocab_eng=vocab_eng, vocab_fra=vocab_fra)
        train_data, val_data = train_test_split(data, test_size=split_val_ratio)
        with open(train_data_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f)
        with open(val_data_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f)
    else:
        with open(train_data_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(val_data_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
    return train_data, val_data


if __name__ == "__main__":
    (vocab_eng, vocab_fra) = create_vocab(vocab_file_eng="vocab_eng.json", vocab_file_fra="vocab_fra.json")
    get_train_val_data(data_file="./data/eng-fra.txt", vocab_eng=vocab_eng, vocab_fra=vocab_fra)
