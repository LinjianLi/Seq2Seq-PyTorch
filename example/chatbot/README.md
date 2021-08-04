# chatbot

## Data Preparation

Download data from [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) or [Cornell Movie-Dialogs Corpus ZIP File](http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip).
Then unzip the file in to a directory called `data`.
The directory structure should look like:

```shell
chatbot
|-- some other files...
|-- data
|  |-- cornell movie-dialogs corpus
|  |  |-- movie_lines.txt
|  |  |-- some other files...
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
