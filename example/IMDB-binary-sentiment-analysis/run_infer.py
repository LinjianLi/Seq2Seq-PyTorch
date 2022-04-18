from model_seq_binary_classifier import SeqBinaryClassifier

from seq2seq.inputter.dataset import Dataset
from seq2seq.inputter.vocab import Vocab

import os
import time
import logging
import json
import argparse
import torch


# The letter "T" is a delimiter suggested in ISO-8601.
# The colon ":" is replaced by the period "." for the log file name.
logging.basicConfig(filename="./log-{}.log".format(
                        time.strftime("%Y-%m-%dT%H.%M.%S", time.gmtime())),
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                    level=logging.INFO)

logger = logging.getLogger(__name__)

logger.info("Program starts with PID: {}".format(os.getpid()))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="./config.json", type=str)
parser.add_argument("--checkpoint", default=None, type=str)
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
config_file = args.config
with open(config_file) as f:
    config = json.load(f)

logger.info("Use GPU: {}.".format(use_gpu))
logger.info("Configurations:\n{}".format(str(config)))

my_vocab = Vocab.from_json("my_vocab.json")

def prepare_data(data_path, vocab):
    data_ids = []
    with open(data_path, "r", encoding="utf-8") as f:
        print("Prepare Data")
        for line in f:
            line = line.split(' ', 1)
            tgt, inp = line[0], line[1]
            tgt = int(tgt)
            inp = inp.lower()
            data_ids.append({"input": vocab.indexes_from_sentence(inp, add_eos=False),
                             "target": tgt})
    return data_ids

test_data = prepare_data("./imdb-binary_sentiment_classification-preprocessed_data/imdb.binary_sentiment_classification.test.txt", my_vocab)

test_data = Dataset(test_data)
test_data_batches = test_data.create_batches(batch_size=config["train_batch_size"], shuffle=False, device=device)

# Load model if a args.checkpoint is provided
if args.checkpoint is not None:
    logger.info('Loading checkpoint file [{}].'.format(args.checkpoint))
    # If loading on same machine the model was trained on
    checkpoint = torch.load(args.checkpoint)
    # If loading a model trained on GPU to CPU
    model_sd = checkpoint['model']
else:
    logger.warning("No checkpoint file provided! Using randomly initialized model!")

logger.info('Building model.')
model = SeqBinaryClassifier(
    src_vocab_size=len(my_vocab),
    embed_size=config["embed_size"],
    hidden_size=config["hidden_size"],
    pretrained_embedding=None,
    batch_first=config["batch_first"],
    num_layers=config["num_layers"],
    bidirectional=config["bidirectional"],
    with_bridge=config["with_bridge"],
    dropout=config["dropout"],
    embedding_dropout=config["embedding_dropout"],
    rnn_cell=config["rnn_cell"],
    use_gpu=use_gpu
)

logger.info(model)

if args.checkpoint is not None:
    logger.info('Loading model state dictionaries.')
    model.load_state_dict(model_sd)
model.to(device)

logger.info('Models built and ready to go!')

all_targets = []
all_preds = []

for batch in test_data_batches:
    preds = model.infer(batch)
    all_targets += batch["target"][0].tolist()
    all_preds += preds.tolist()

correct = [1 if all_targets[i] == all_preds[i] else 0 for i in range(len(all_targets))]

acc_str = "Accuracy: {}".format(sum(correct) / len(correct))

print(acc_str)
logger.info(acc_str)

logger.info("Inference finished.")
