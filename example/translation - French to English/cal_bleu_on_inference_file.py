from nltk.translate.bleu_score import corpus_bleu
import argparse
import time
import logging

logging.basicConfig(filename="./log-{}.log".format(time.strftime('%Y-%m-%d %H.%M.%S', time.gmtime())),
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)

parser = argparse.ArgumentParser()
parser.add_argument("--inference_file", default="./inference.txt", type=str)
args = parser.parse_args()

def cal_bleu(file):
    logger.info(file)
    with open(file, "r") as f:
        references, candidates = [], []
        for line in f:
            if line.startswith("target:"):
                references.append(line.split("\t")[-1].split())
            elif line.startswith("infer:"):
                candidates.append(line.split("\t")[-1].split())
        logger.info("Corpus BLEU*100: {}"\
                        .format(corpus_bleu(references, candidates) * 100))

if __name__ == "__main__":
    cal_bleu(args.inference_file)
