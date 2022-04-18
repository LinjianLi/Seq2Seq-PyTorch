# 2022-04-16

import torch
from torchtext import data
from torchtext import datasets
import random

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm')
LABEL = data.LabelField(dtype = torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state = random.seed(SEED))

def write_split(data_split, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for instance in list(map(vars, data_split.examples)):
            sentiment = 1 if instance["label"] == "pos" else 0
            sentence = " ".join(instance["text"])
            f.write("{} {}\n".format(sentiment, sentence))
    print("Saved [{}]".format(file_name))

write_split(train_data, "imdb.binary_sentiment_classification.train.txt")
write_split(valid_data, "imdb.binary_sentiment_classification.valid.txt")
write_split(test_data, "imdb.binary_sentiment_classification.test.txt")