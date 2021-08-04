import os
import logging
import re
import json
import csv
import codecs
from sklearn.model_selection import train_test_split
from progress_text import ProgressText
from seq2seq.inputter.vocab import Vocab

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
# Since there are a lot of example sentences and we want to train something quickly,
# we’ll trim the data set to only relatively short and simple sentences.

MAX_LENGTH = 15


def normalize_str(s):
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def normalizePair(p):
    p = [normalize_str(s) for s in p]
    p = [s.lower() for s in p]
    return p


def pair_is_simple(p):
    return (len(p[0].split(' ')) < MAX_LENGTH
            and
            len(p[1].split(' ')) < MAX_LENGTH)
# ----------------------------------------------------------


def create_formatted_data_file():

    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join("data", corpus_name)

    # Splits each line of the file into a dictionary of fields
    def loadLines(fileName, fields):
        lines = {}
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj
        return lines


    # Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
    def loadConversations(fileName, lines, fields):
        conversations = []
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]
                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                utterance_id_pattern = re.compile('L[0-9]+')
                lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
                # Reassemble lines
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(lines[lineId])
                conversations.append(convObj)
        return conversations


    # Extracts pairs of sentences from conversations
    def extractSentencePairs(conversations):
        qa_pairs = []
        for conversation in conversations:
            # Iterate over all the lines of the conversation
            for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
                inputLine = conversation["lines"][i]["text"].strip()
                targetLine = conversation["lines"][i+1]["text"].strip()
                # Filter wrong samples (if one of the lists is empty)
                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])
        return qa_pairs

    # Define path to new file
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict, conversations list, and field ids
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                    lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)


def create_vocab(vocab_file_eng):
    if not os.path.exists(vocab_file_eng):
        vocab_eng = Vocab("eng")
        logger.info('Creating vocab.')
        with open("./data/cornell movie-dialogs corpus/formatted_movie_lines.txt", "r", encoding="utf-8") as f:
            for line in ProgressText(f.readlines(), task_name="Create Vocab"):
                line = line.split('\t')
                line = normalizePair(line)
                if not pair_is_simple(line):
                    continue
                vocab_eng.add_sentence(line[0], to_lower=True, remove_punc=False)
                vocab_eng.add_sentence(line[1], to_lower=True, remove_punc=False)
        logger.info(vocab_eng)
        logger.info('Storing vocab.')
        vocab_eng.to_json(vocab_file_eng)
        return vocab_eng


def get_vocab():
    vocab_file_eng = "vocab_eng.json"
    if not os.path.exists(vocab_file_eng):
        return create_vocab(vocab_file_eng)
    else:
        logger.info('Loading vocab.')
        vocab_eng = Vocab.from_json(vocab_file_eng)
        logger.info(vocab_eng)
        return vocab_eng


def prepare_data(data_path, vocab_eng):
    data_ids = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in ProgressText(f.readlines(), task_name="Prepare Data"):
            line = line.split('\t')
            line = normalizePair(line)
            if not pair_is_simple(line):
                continue
            inp, tgt = line[0], line[1]
            data_ids.append({"input": vocab_eng.indexes_from_sentence(inp, add_eos=True),
                             "target": vocab_eng.indexes_from_sentence(tgt, add_eos=True)})
    return data_ids


def get_train_val_data(data_file="./data/cornell movie-dialogs corpus/formatted_movie_lines.txt", vocab_eng=None):
    logger.info('Preparing data.')
    train_data_path = 'train_data.json'
    val_data_path = 'val_data.json'
    test_data_path = 'test_data.json'
    # train : valid : test = 8 : 1 : 1
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path) or not os.path.exists(test_data_path):
        data = prepare_data(data_file, vocab_eng=vocab_eng)
        train_data, val_test_data = train_test_split(data, test_size=0.2)
        val_data, test_data = train_test_split(val_test_data, test_size=0.5)
        with open(train_data_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f)
        with open(val_data_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f)
        with open(test_data_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
    else:
        with open(train_data_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(val_data_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    return train_data, val_data, test_data


if __name__ == "__main__":
    create_formatted_data_file()
    vocab_eng = get_vocab()
    get_train_val_data(data_file="./data/cornell movie-dialogs corpus/formatted_movie_lines.txt", vocab_eng=vocab_eng)
