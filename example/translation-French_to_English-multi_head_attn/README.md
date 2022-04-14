# translation-French_to_English

## Data Preparation

Download French-English translation data from [Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/) or [Tab-delimited Bilingual Sentence Pairs fra-eng.zip ZIP File](http://www.manythings.org/anki/fra-eng.zip).
Then unzip the file in to a directory called `data` and rename the file as `eng-fra.txt` (because the first colomn in the file is in English and the second is in French, and it is written in the logic of the example code).
The directory structure should look like:

```shell
translation-French_to_English
|-- some other files...
|-- data
|  |-- eng-fra.txt
|  |-- some other files...
```

## Run Example

Please install `seq2seq-pytorch` or copy the `seq2seq` folder to the same directory as the `run_train.py` is in.

Execute:

```shell
python prepare_vocab_and_data.py
python run_train.py
python run_infer.py --checkpoint "replace_with_the_checkpoint_file_you_want"
python cal_bleu_on_inference_file.py -f "replace_with_the_inference_file_you_want"
```
