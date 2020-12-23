# translation - French to English

Please install `seq2seq-pytorch` or copy the `seq2seq` folder to the same directory as the `run_train.py` is in.

First

```shell
python run_train.py
```

Then

```shell
python run_infer.py --checkpoint "replace_with_the_checkpoint_file_you_want"
```

Last

```shell
python cal_bleu_on_inference_file.py
```
