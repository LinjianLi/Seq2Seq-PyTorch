from nltk.translate.bleu_score import corpus_bleu
import argparse
import time
import logging

# The letter "T" is a delimiter suggested in ISO-8601.
# The colon ":" is replaced by the period "." for the log file name.
logging.basicConfig(filename="./log-{}.log".format(
                        time.strftime("%Y-%m-%dT%H.%M.%S", time.gmtime())),
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", default="./inference.txt", type=str)
args = parser.parse_args()


def cal_bleu(file):
    logger.info(file)
    with open(file, "r") as f:
        references, candidates = [], []
        for line in f:
            if line.startswith("target:"):
                references.append([line.split("\t")[-1].split()])
            elif line.startswith("infer:"):
                candidates.append(line.split("\t")[-1].split())
        logger.info("Corpus (1-to-4 Gram) BLEU*100: {}"\
                        .format(corpus_bleu(references, candidates) * 100))
        logger.info("Corpus (1-to-3 Gram) BLEU*100: {}"\
                        .format(corpus_bleu(references, candidates, weights=[1/3, 1/3, 1/3, 0]) * 100))
        logger.info("Corpus (1-to-2 Gram) BLEU*100: {}"\
                        .format(corpus_bleu(references, candidates, weights=[1/2, 1/2, 0, 0]) * 100))
        logger.info("Corpus (1 Gram) BLEU*100: {}"\
                        .format(corpus_bleu(references, candidates, weights=[1, 0, 0, 0]) * 100))


if __name__ == "__main__":
    cal_bleu(args.file)
