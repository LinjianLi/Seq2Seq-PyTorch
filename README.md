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

Or simply copy the source code.

## Usage

Install or copy `seq2seq` folder to the project directory as a package before using.
See [example](./example/translation%20-%20French%20to%20English)

## To Do

- Fix `trainer`. When saving training checkpoint, `trainer` does not save the best epoch model. So, if resume training, the saved best epoch after finishing is not actually the best epoch of the whole training stage, but the best epoch after the checkpoint.
- Support [gradient accumulation](https://ai.stackexchange.com/questions/21972/what-is-the-relationship-between-gradient-accumulation-and-batch-size) which enables greater batch size although with limited memory.
- (Not sure if it is necessary.) Support regression.
