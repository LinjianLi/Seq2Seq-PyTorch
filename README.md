# Seq2Seq-PyTorch

Sequence-to-sequence implementation using PyTorch

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
See [example](./example/translation%20-%20French%20to%20English)

## Some Features

- `Trainer` supports [gradient accumulation](https://ai.stackexchange.com/questions/21972/what-is-the-relationship-between-gradient-accumulation-and-batch-size) which enables larger (equivalent) batch size although with limited memory.

## To Do

- Support beam search.
- Fix `trainer`. When saving training checkpoint, `trainer` does not save the best epoch model. So, if resume training, the saved best epoch after finishing is not actually the best epoch of the whole training stage, but the best epoch after the checkpoint. (Not sure if trainer should save the best-so-far model at every checkpoint, which will make the checkpoint file large.)
- (Not sure if it is necessary.) Support regression.
- Refactor the structure of files in this project.
- Add some utility scripts, such as `create_vocab.py`, `inference.py`, and so on.
- Add evaluater and use evaluater in trainer.
