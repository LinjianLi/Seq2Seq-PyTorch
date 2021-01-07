import os
import time
import json
import argparse
import random
import re

import torch
from torch import optim
from sklearn.model_selection import train_test_split

from progress_text import ProgressText

from seq2seq.criterions import NLLLoss
from seq2seq.dataset import Dataset
from seq2seq.vocab import Vocab
from seq2seq.seq2seq import Seq2Seq
from seq2seq.trainer import Trainer

import logging

logging.basicConfig(filename="./log-{}.log".format(time.strftime('%Y-%m-%d %H.%M.%S', time.gmtime())),
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)

logger.info("Program starts with PID: {}".format(os.getpid()))

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
config_file = "./config.json"
with open(config_file) as f:
    config = json.load(f)

logger.info("Use GPU: {}.".format(use_gpu))
logger.info("Configurations:\n{}".format(str(config)))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, type=str)
args = parser.parse_args()

#----------------------------------------------------------
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Since there are a lot of example sentences and we want to train something quickly,
# we’ll trim the data set to only relatively short and simple sentences.

MAX_LENGTH = config["max_length"]

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
    # If pair is simple, it will be keeped. return True
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[0].startswith(accept_eng_prefixes)
#----------------------------------------------------------

vocab_file_eng = "vocab_eng.json"
vocab_file_fra = "vocab_fra.json"
if not os.path.exists(vocab_file_eng) or not os.path.exists(vocab_file_fra):
    vocab_eng = Vocab("eng")
    vocab_fra = Vocab("fra")
    logger.info('Creating vocab.')
    with open("./data/eng-fra.txt", "r") as f:
        for line in ProgressText(f.readlines()):
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
else:
    logger.info('Loading vocab.')
    vocab_eng = Vocab("eng")
    vocab_fra = Vocab("fra")
    vocab_eng.from_json(vocab_file_eng)
    vocab_fra.from_json(vocab_file_fra)
    logger.info(vocab_eng)
    logger.info(vocab_fra)

def prepare_data(data_path, vocab_eng, vocab_fra):
    data_ids = []
    with open(data_path, "r") as f:
        for line in ProgressText(f.readlines()):
            line = line.split('\t')
            line = normalizePair(line)
            if not pair_is_simple(line):
                continue
            tgt, inp = line # French to English
            data_ids.append({"input": vocab_fra.indexes_from_sentence(inp, add_sos=True, add_eos=True),
                             "target": vocab_eng.indexes_from_sentence(tgt, add_sos=True, add_eos=True)})
    return data_ids

logger.info('Preparing data.')
train_data_path = 'train_data.json'
val_data_path = 'val_data.json'
split_val_ratio = 0.2
if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
    data = prepare_data("./data/eng-fra.txt", vocab_eng=vocab_eng, vocab_fra=vocab_fra)
    train_data, val_data = train_test_split(data, test_size=split_val_ratio)
    with open(train_data_path, "w") as f:
        json.dump(train_data, f)
    with open(val_data_path, "w") as f:
        json.dump(val_data, f)
else:
    with open(train_data_path, "r") as f:
        train_data = json.load(f)
    with open(val_data_path, "r") as f:
        val_data = json.load(f)

train_data = Dataset(train_data)
train_data_batches = train_data.create_batches(batch_size=config["train_batch_size"], shuffle=False, device=device)
val_data = Dataset(val_data)
val_data_batches = val_data.create_batches(batch_size=config["eval_batch_size"], shuffle=False, device=device)

logger.info('Building mode.')
model = Seq2Seq(src_vocab_size=len(vocab_fra),
                tgt_vocab_size=len(vocab_eng),
                embed_size=config["embed_size"],
                hidden_size=config["hidden_size"],
                padding_idx=config["PAD_token"],
                batch_first=config["batch_first"],
                num_layers=config["num_layers"],
                bidirectional=config["bidirectional"],
                attn_mode=config["attn_mode"],
                attn_hidden_size=config["attn_hidden_size"],
                with_bridge=config["with_bridge"],
                tie_embedding=config["tie_embedding"],
                dropout=config["dropout"],
                rnn_cell=config["rnn_cell"],
                teacher_forcing_ratio=config["teacher_forcing_ratio"],
                use_gpu=use_gpu)

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

if config["scheduler"].lower() == "steplr":
    optimizer_scheduler = optim.lr_scheduler.StepLR(optimizer=model_optimizer,
                                                    step_size=config["scheduler_step_size"],
                                                    gamma=config["scheduler_gamma"],
                                                    last_epoch=-1)
else:
    optimizer_scheduler = None

save_dir = config["save_dir"]

trainer = Trainer(model=model,
                  loss_fn=NLLLoss(ignore_index=config["PAD_token"], reduction='mean'),
                  optimizer=model_optimizer,
                  scheduler=optimizer_scheduler,
                  num_epoches=config["num_epoches"],
                  gradient_accumulation=config["gradient_accumulation"],
                  train_dataloder=train_data_batches,
                  valid_dataloder=val_data_batches,
                  early_stop_num=config["early_stop_num"],
                  save_path=config["save_dir"],
                  save_every_epoch=config["save_every_epoch"],
                  plot_loss_group_by=config["plot_loss_group_by"],
                  plot_loss_group_by_every=config["plot_loss_group_by_every"],
                  evaluate_before_train=config["evaluate_before_train"])

if args.checkpoint is not None:
    trainer.load(args.checkpoint)

# Run training iterations
logger.info("Start training.")
trainer.run(grad_clip=config["grad_clip"], progress_indicator=config["progress_indicator"])
logger.info("Finish training.")

logger.info("Some examples from the final model:\n")
for data in random.choices(val_data, k=5):
    input, target = data["input"], data["target"]
    infer = model.infer(input, start_token=target[0])

    input = " ".join(vocab_fra.sentence_from_indexes(input)).replace("<EOS> ", "")
    infer = " ".join(vocab_eng.sentence_from_indexes(infer)).replace("<EOS> ", "")
    target = " ".join(vocab_eng.sentence_from_indexes(target)).replace("<EOS> ", "")
    string = "\n" + "input:\t" + input + "\n" + "infer:\t" + infer + "\n" + "target:\t" + target
    logger.info(string)
