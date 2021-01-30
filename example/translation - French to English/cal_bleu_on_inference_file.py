from nltk.translate.bleu_score import corpus_bleu
import argparse
import time
import logging

logging.basicConfig(filename="./log-{}.log".format(time.strftime('%Y-%m-%d %H.%M.%S', time.gmtime())),
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--inference_file", default="./inference.txt", type=str)
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
    cal_bleu(args.inference_file)
