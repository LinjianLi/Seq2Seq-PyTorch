from model_seq_binary_classifier import SeqBinaryClassifier

from seq2seq.inputter.dataset import Dataset
from seq2seq.trainer.trainer import Trainer
from seq2seq.inputter.vocab import Vocab

import os
import time
import logging
import json
import argparse
import torch
from torch import optim
import torch.nn.functional as F


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

datalist = [
    "./imdb-binary_sentiment_classification-preprocessed_data/imdb.binary_sentiment_classification.train.txt",
    "./imdb-binary_sentiment_classification-preprocessed_data/imdb.binary_sentiment_classification.valid.txt",
    "./imdb-binary_sentiment_classification-preprocessed_data/imdb.binary_sentiment_classification.test.txt"
]

vocab_file = "my_vocab.json"
if not os.path.exists(vocab_file):
    my_vocab = Vocab("my_vocab")
    for data in datalist:
        with open(data, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split(' ', 1)
                tgt, inp = line[0], line[1]
                my_vocab.add_sentence(inp, to_lower=True, remove_punc=False)
    logger.info("vocab size: {}".format(len(my_vocab)))
    my_vocab.keep_most_frequent_k(k=50000)
    my_vocab.to_json("my_vocab.json")
else:
    logger.info('Loading vocab...')
    my_vocab = Vocab.from_json(vocab_file)

pretrain_embedding = my_vocab.extract_pretrain_embedding("./glove.6B.100d.txt", 100)

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

train_data, val_data, test_data = [prepare_data(data, my_vocab) for data in datalist]

train_data = Dataset(train_data)
train_data_batches = train_data.create_batches(batch_size=config["train_batch_size"], shuffle=True, device=device)
val_data = Dataset(val_data)
val_data_batches = val_data.create_batches(batch_size=config["eval_batch_size"], shuffle=False, device=device)
test_data = Dataset(test_data)

logger.info('Building model.')
model = SeqBinaryClassifier(
    src_vocab_size=len(my_vocab),
    embed_size=config["embed_size"],
    hidden_size=config["hidden_size"],
    pretrained_embedding=pretrain_embedding,
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
logger.info('Models built and ready to go!')

# Initialize optimizers
logger.info('Building optimizers.')
if config["optimizer"].lower() == "adam":
    model_optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
elif config["optimizer"].lower() == "sgd":
    model_optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
else:
    model_optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

# Initialize optimizer scheduler.
if config["scheduler"].lower() == "steplr":
    optimizer_scheduler = optim.lr_scheduler.StepLR(optimizer=model_optimizer,
                                                    step_size=config["scheduler_step_size"],
                                                    gamma=config["scheduler_gamma"],
                                                    last_epoch=-1)
elif config["scheduler"].lower() != "none":
    logger.warning("Scheduler {} not supported yet. Not using scheduler.".format(config["scheduler"]))
    optimizer_scheduler = None
else:
    optimizer_scheduler = None

trainer = Trainer(
    model=model,
    loss_fn=F.nll_loss,
    optimizer=model_optimizer,
    scheduler=optimizer_scheduler,
    num_epochs=config["num_epochs"],
    gradient_accumulation=config["gradient_accumulation"],
    train_dataloader=train_data_batches,
    valid_dataloader=val_data_batches,
    early_stop_num=config["early_stop_num"],
    save_best_model=config["save_best_model"],
    save_path=config["save_dir"],
    save_every_epoch=config["save_every_epoch"],
    plot_loss_group_by=config["plot_loss_group_by"],
    plot_loss_group_by_every=config["plot_loss_group_by_every"],
    evaluate_before_train=config["evaluate_before_train"],
    use_gpu=use_gpu
)

if args.checkpoint is not None:
    trainer.load(args.checkpoint)

# Run training iterations
logger.info("Start training.")
trainer.run(grad_clip=config["grad_clip"])
logger.info("Finish training.")
