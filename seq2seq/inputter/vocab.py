import os
import json
import re
import logging

logger = logging.getLogger(__name__)

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unknown token


class Vocab:
    def __init__(self, name="unnamed"):
        self.name = name
        self.trimmed = False
        self.word2index = {"<PAD>": PAD_token, "<SOS>": SOS_token,
                           "<EOS>": EOS_token, "<UNK>": UNK_token}
        self.index2word = {PAD_token: "<PAD>", SOS_token: "<SOS>",
                           EOS_token: "<EOS>", UNK_token: "<UNK>"}
        self.word2count = {}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK

    def to_json(self, filename, encoding="utf-8"):
        data = {"name": self.name,
                "trimmed": self.trimmed,
                "word2index": self.word2index,
                # "index2word": self.index2word,
                "word2count": self.word2count,
                "num_words": self.num_words}
        with open(filename, mode="w", encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("{} dumped to file {}".format(self, filename))

    @classmethod
    def from_json(cls, filename, encoding="utf-8"):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        with open(filename, mode="r", encoding=encoding) as f:
            data = json.load(f)
        voc = Vocab()
        voc.name = data["name"]
        voc.trimmed = data["trimmed"]
        voc.word2index = data["word2index"]
        # voc.index2word = data["index2word"]
        voc.word2count = data["word2count"]
        voc.num_words = data["num_words"]

        voc.index2word = {}
        for word in voc.word2index:
            voc.index2word[voc.word2index[word]] = word

        logger.info("{} loaded from file {}".format(voc, filename))
        return voc

    def __len__(self):
        return self.num_words

    def __repr__(self):
        return "Vocab [{}] of size {}".format(self.name, self.num_words)

    def add_sentences(self, sentences, to_lower=False, remove_punc=False):
        for sentence in sentences:
            self.add_sentence(sentence, to_lower, remove_punc)

    def add_sentence(self, sentence, to_lower=False, remove_punc=False):
        # Make sure the sentence is a string.
        # If it is a list of strings (words), convert it to a single string.
        assert isinstance(sentence, (str, list))
        if isinstance(sentence, list):
            assert isinstance(sentence[0], str)
            sentence = " ".join(sentence)

        sentence = re.sub(r"(\s\t\n+)", r" ", sentence)
        sentence = sentence.strip()
        if to_lower:
            sentence = sentence.lower()
        if remove_punc:
            sentence = re.sub(r"([,.!?'\"])", r" ", sentence)
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def get_index(self, word):
        # If the word is unseen, return the unknown token.
        return self.word2index.get(word, self.word2index["<UNK>"])

    def get_word(self, index):
        return self.index2word.get(index, self.index2word[UNK_token])

    # Remove words below a certain count threshold
    def trim(self, min_count=10):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append((k, v))

        logger.info('keep_words {} / {} = {:.4f}'.format(
                        len(keep_words),
                        len(self.word2index),
                        len(keep_words) / len(self.word2index)))

        # Reinitialize dictionaries
        self.word2index = {"<PAD>": PAD_token, "<SOS>": SOS_token,
                           "<EOS>": EOS_token, "<UNK>": UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: "<PAD>", SOS_token: "<SOS>",
                           EOS_token: "<EOS>", UNK_token: "<UNK>"}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK

        for (word, count) in keep_words:
            self.add_word(word)
            self.word2count[word] = count

    # Keep the most frequent k words.
    def keep_most_frequent_k(self, k=10000):
        sorted_word_count = list(self.word2count.items())
        sorted_word_count.sort(key=lambda w_c: w_c[1], reverse=True)
        sorted_word_count = sorted_word_count[0:k]

        logger.info('keep_words {} / {} = {:.4f}'.format(
                        len(sorted_word_count),
                        len(self.word2index),
                        len(sorted_word_count) / len(self.word2index)))

        # Reinitialize dictionaries
        self.word2index = {"<PAD>": PAD_token, "<SOS>": SOS_token,
                           "<EOS>": EOS_token, "<UNK>": UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: "<PAD>", SOS_token: "<SOS>",
                           EOS_token: "<EOS>", UNK_token: "<UNK>"}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK

        for (word, count) in sorted_word_count:
            self.add_word(word)
            self.word2count[word] = count

    def indexes_from_sentence(self, sentence, add_sos=False, add_eos=False):
        assert (len(sentence) > 0), ("The input is empty!")
        if isinstance(sentence, str):
            sentence = re.sub(r"(\s\t\n+)", r" ", sentence)
            sentence = sentence.strip().split()
        indexes = [self.get_index(word) for word in sentence]
        if add_sos:
            indexes = [self.get_index("<SOS>")] + indexes
        if add_eos:
            indexes.append(self.get_index("<EOS>"))
        return indexes

    def sentence_from_indexes(self, indexes):
        assert (len(indexes) > 0), ("The input is empty!")
        assert isinstance(indexes, (list, tuple))
        assert isinstance(indexes[0], int)
        sentence = [self.get_word(index) for index in indexes]
        return sentence
