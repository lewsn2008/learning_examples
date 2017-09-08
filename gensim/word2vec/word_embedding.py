#!/usr/bin/env python
#coding=utf8
# author: Andy Liu
from __future__ import print_function
import argparse
import logging
import sys

from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import Text8Corpus

reload(sys)
sys.setdefaultencoding("utf8")

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format=('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d :'
            '%(message)s'), datefmt='%a, %d %b %Y %H:%M:%S')

class Sentences(object):
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file

    def __iter__(self):
        for line in open(self.corpus_file):
            line = line.strip().lower()
            if not isinstance(line, unicode):
                try:
                    line = line.decode('utf8')
                except:
                    logging.error('Failed to decode line %s' % line, file=sys.stderr)

            yield line.split('\t')


def train(args):
    if not args.corpus_file:
        logging.error('Corpus path needed!')
        sys.exit(1)

    sentences = Sentences(args.corpus_file)
    #sentences = Text8Corpus(args.corpus_file)
    model = Word2Vec(sentences, min_count=5, size=args.dimension)
    logging.info('Finished training!')

    return model


def restore(args):
    if not args.model:
        logging.error('Model path needed!')
        sys.exit(1)

    return Word2Vec.load(args.model)


def eval(model):
    while True:
        word = raw_input('Input word : ')
        if not word:
            break

        if not isinstance(word, unicode):
            word = word.decode('utf8')

        print('====== Test of word: %s' % word)
        if word not in model.wv.vocab:
            print('Word not exist in vocabulary!', file=sys.stderr)
            continue

        #print('Embedding vector :')
        #print(model.wv[word])

        print('Top 10 most simiar words:')
        l = model.most_similar(positive=[word], topn=10)
        for (word, v) in l:
            print('%s\t%f' % (word, v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_file', type=str, help='Location of corpus file.')
    parser.add_argument('--dimension', type=int, default=100, help='Vector dimension')
    parser.add_argument('--model_file', type=str, help='Location of model file.')
    args = parser.parse_args()

    # Train model from corpus.
    if args.corpus_file:
        model = train(args)
        # Save model.
        if args.model_file:
            model.save(args.model_file)
    # Load model from file.
    elif args.model_file:
        model = restore(args)
    else:
        logging.error('Wrong usage!')
        sys.exit(1)

    # Evaluation of inputs.
    eval(model)

