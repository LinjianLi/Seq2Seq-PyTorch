# IMDB Binary Sentiment Analysis

Sentiment analysis using [IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/).

Although it not a sequence-to-sequence task, it can be used to verify the effectiveness of some modules in our `seq2seq` framework. Such as, `Trainer`, `Vocab`, `SimpleRNN`, `utils`, etc..

## Usage

Install Python package `scapy` and the correspoinding model `en_core_web_sm`.

```shell
conda install scapy
python -m scapy download en_core_web_sm
```

Go to the `imdb-binary_sentiment_classification-preprocessed_data` directory, and then run `python get_data.py` to get the preprocessed data.

```shell
cd imdb-binary_sentiment_classification-preprocessed_data
python get_data.py
```

Then run `run_train.py` to train the model. Run `run_infer.py` with argument `--checkpoint <your_checkpoint>` to test the trained model.

## Performance

Under the settings in `config.json`, the test accuracy of the best model should be around 85%.
