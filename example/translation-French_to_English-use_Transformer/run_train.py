import os
import time
import logging
import json
import argparse
import random
import torch
from torch import optim

from seq2seq.criterion.nll_loss import NLLLoss
from seq2seq.inputter.dataset import Dataset
from seq2seq.model.seq2seq_transformer import Seq2SeqTransformer
from seq2seq.trainer.trainer import Trainer

from prepare_vocab_and_data import get_vocab, get_train_val_data


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

(vocab_eng, vocab_fra) = get_vocab()

(train_data, val_data) = get_train_val_data(data_file="./data/eng-fra.txt", vocab_eng=vocab_eng, vocab_fra=vocab_fra)

train_data = Dataset(train_data)
train_data_batches = train_data.create_batches(batch_size=config["train_batch_size"], shuffle=True, device=device)
val_data = Dataset(val_data)
val_data_batches = val_data.create_batches(batch_size=config["eval_batch_size"], shuffle=False, device=device)

logger.info('Building model.')
model = Seq2SeqTransformer(
    src_vocab_size=len(vocab_fra),
    tgt_vocab_size=len(vocab_eng),
    embed_size=config["embed_size"],
    hidden_size=config["hidden_size"],
    start_token=config["SOS_token"],
    end_token=config["EOS_token"],
    padding_idx=config["PAD_token"],
    batch_first=config["batch_first"],
    num_layers=config["num_layers"],
    attn_head_num=config["attn_head_num"],
    tie_embedding=config["tie_embedding"],
    dropout=config["dropout"],
    teacher_forcing_ratio=config["teacher_forcing_ratio"],
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
    loss_fn=NLLLoss(ignore_index=config["PAD_token"], reduction='mean'),
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
    use_gpu=use_gpu,
    config=config,
)

if args.checkpoint is not None:
    trainer.load(args.checkpoint)

# Run training iterations
logger.info("Start training.")
trainer.run(grad_clip=config["grad_clip"])
logger.info("Finish training.")

logger.info("Some examples from the final model:\n")
for data in random.choices(val_data, k=5):
    inp, target = data["input"], data["target"]
    infer = model.infer(inp, max_length=20)

    inp = " ".join(vocab_fra.sentence_from_indexes(inp))
    infer = " ".join(vocab_eng.sentence_from_indexes(infer))
    target = " ".join(vocab_eng.sentence_from_indexes(target))
    string = "\n" + "input:\t" + inp + "\n" + "infer:\t" + infer + "\n" + "target:\t" + target
    logger.info(string)
