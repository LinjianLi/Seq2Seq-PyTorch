# Seq2Seq-PyTorch

Sequence-to-sequence models implementation using PyTorch framework.

Note: I reuse many source codes written by others. This repository is just for my daily practice, instead of any commercial use.

## Install

Clone the project, go into the project directory, and execute

```shell
python setup.py install
```

or

```shell
pip install ./
```

or simply copy the source code.

`pip install ./` is recommended, because you can activate a virtual environment first and then install the package in that environment without affecting others.

## Usage

Install or copy `seq2seq` folder to the project directory as a package before using.
See [example](./example/translation-French_to_English)

## Some Features

- `Trainer` supports [gradient accumulation](https://ai.stackexchange.com/questions/21972/what-is-the-relationship-between-gradient-accumulation-and-batch-size) which enables larger (equivalent) batch size although with limited memory.
- Supports beam search by reusing the code from the [Transformers library of the HuggingFace Inc. team](https://github.com/huggingface/transformers).
- Attention mechanism. I use the [Luong attention mechanism](https://arxiv.org/abs/1508.04025) instead of the [Bahdanau attention mechanism](https://arxiv.org/abs/1409.0473). When computing the attention at time step `t`, the former uses the hidden state from the time step `t` while the latter uses the hidden state from the time step `t-1`.

## To Do

- Fix `trainer`. When saving training checkpoint, `trainer` does not save the best epoch model. So, if resume training, the saved best epoch after finishing is not actually the best epoch of the whole training stage, but the best epoch after the checkpoint. (Not sure if trainer should save the best-so-far model at every checkpoint, which will make the checkpoint file large.)
- (Not sure if it is necessary.) Support regression.
- Add some utility scripts, such as `create_vocab.py`, `inference.py`, and so on.
