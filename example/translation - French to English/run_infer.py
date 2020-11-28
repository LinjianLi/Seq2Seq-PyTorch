import torch
import os
import json
import argparse
import random

from seq2seq.vocab import Vocab
from seq2seq.seq2seq import Seq2Seq

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
with open("./config.json") as f:
    config = json.load(f)

logger.info("Use GPU: {}.".format(use_gpu))
logger.info("Configurations:\n{}".format(str(config)))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, type=str)
args = parser.parse_args()

vocab_file_eng = "vocab_eng.json"
if not os.path.exists(vocab_file_eng):
    raise FileNotFoundError
else:
    vocab_eng = Vocab("eng")
    logger.info('Loading vocab...')
    vocab_eng.from_json(vocab_file_eng)

vocab_file_fra = "vocab_fra.json"
if not os.path.exists(vocab_file_fra):
    raise FileNotFoundError
else:
    vocab_fra = Vocab("fra")
    logger.info('Loading vocab...')
    vocab_fra.from_json(vocab_file_fra)

logger.info('Preparing data...')
val_data_path = 'val_data.json'
if not os.path.exists(val_data_path):
    raise FileNotFoundError
else:
    with open(val_data_path, "r") as f:
        val_data = json.load(f)

# Load model if a args.checkpoint is provided
if args.checkpoint is not None:
    logger.info('Loading checkpoint file [{}]...'.format(args.checkpoint))
    # If loading on same machine the model was trained on
    checkpoint = torch.load(args.checkpoint)
    # If loading a model trained on GPU to CPU
    model_sd = checkpoint['model']
else:
    logger.warning("No checkpoint file proveded! Using randomly initialized model!")

logger.info('Building model...')
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
if args.checkpoint is not None:
    logger.info('Loading model state dictionaries...')
    model.load_state_dict(model_sd)
model.to(device)

logger.info('Models built and ready to go!')

for data in random.choices(val_data, k=20):
    input, target = data["input"], data["target"]
    infer = model.infer(input, start_token=target[0])

    input = " ".join(vocab_fra.sentence_from_indexes(input)).replace("<EOS> ", "")
    infer = " ".join(vocab_eng.sentence_from_indexes(infer)).replace("<EOS> ", "")
    target = " ".join(vocab_eng.sentence_from_indexes(target)).replace("<EOS> ", "")
    string = "\n" + "input:\t" + input + "\n" + "infer:\t" + infer + "\n" + "target:\t" + target
    logger.info(string)
