import os
import time
import logging
import json
import argparse
from tqdm import tqdm
import torch

from seq2seq.inputter.vocab import Vocab
from seq2seq.model.seq2seq import Seq2Seq

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
with open(config_file, "r", encoding="utf-8") as f:
    config = json.load(f)

logger.info("Use GPU: {}.".format(use_gpu))
logger.info("Configurations:\n{}".format(str(config)))

vocab_file_eng = "vocab_eng.json"
vocab_file_fra = "vocab_fra.json"
if not os.path.exists(vocab_file_eng):
    raise FileNotFoundError(vocab_file_eng)
elif not os.path.exists(vocab_file_fra):
    raise FileNotFoundError(vocab_file_fra)
else:
    logger.info('Loading vocab.')
    vocab_eng = Vocab.from_json(vocab_file_eng)
    vocab_fra = Vocab.from_json(vocab_file_fra)
    logger.info(vocab_eng)
    logger.info(vocab_fra)

logger.info('Preparing data.')
val_data_path = 'val_data.json'
if not os.path.exists(val_data_path):
    raise FileNotFoundError(val_data_path)
else:
    with open(val_data_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

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
model = Seq2Seq(
    src_vocab_size=len(vocab_fra),
    tgt_vocab_size=len(vocab_eng),
    embed_size=config["embed_size"],
    hidden_size=config["hidden_size"],
    start_token=config["SOS_token"],
    end_token=config["EOS_token"],
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
    use_gpu=use_gpu
)

if args.checkpoint is not None:
    logger.info('Loading model state dictionaries.')
    model.load_state_dict(model_sd)
model.to(device)

logger.info('Models built and ready to go!')

logger.info("Inference greedy search start.")
with open("./inference-greedy_search.txt", mode="w", encoding="utf-8") as f:
    model_info = "Model from checkpoint: {}\n".format(args.checkpoint)
    f.write(model_info)
    for data in tqdm(val_data):
        inp, target = data["input"], data["target"]
        infer = model.infer(inp, max_length=20)

        inp = " ".join(vocab_fra.sentence_from_indexes(inp))
        infer = " ".join(vocab_eng.sentence_from_indexes(infer))
        target = " ".join(vocab_eng.sentence_from_indexes(target))
        string = "\n" + "input:\t" + inp + "\n" + "infer:\t" + infer + "\n" + "target:\t" + target + "\n"
        f.write(string)
logger.info("Inference greedy search finished.")

logger.info("Inference beam search start.")
with open("./inference-beam_search.txt", mode="w", encoding="utf-8") as f:
    model_info = "Model from checkpoint: {}\n".format(args.checkpoint)
    f.write(model_info)
    for data in tqdm(val_data):
        inp, target = data["input"], data["target"]
        infer = model.infer_beam(inp, max_length=20, beam_width=5)

        inp = " ".join(vocab_fra.sentence_from_indexes(inp))
        infer = " ".join(vocab_eng.sentence_from_indexes(infer))
        target = " ".join(vocab_eng.sentence_from_indexes(target))
        string = "\n" + "input:\t" + inp + "\n" + "infer:\t" + infer + "\n" + "target:\t" + target + "\n"
        f.write(string)
logger.info("Inference beam search finished.")

logger.info("Inference finished.")
