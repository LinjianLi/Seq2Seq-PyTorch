# chatbot

Please install `seq2seq-pytorch` or copy the `seq2seq` folder to the same directory as the `run_train.py` is in.

Execute:

```shell
python prepare_vocab_and_data.py
python run_train.py
python run_infer.py --checkpoint "replace_with_the_checkpoint_file_you_want"
python cal_bleu_on_inference_file.py -f "replace_with_the_inference_file_you_want"
```
