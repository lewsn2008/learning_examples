#!/usr/bin/env python
#coding=utf8
from __future__ import print_function
import argparse
import logging
import sys

from gensim.corpora import Dictionary
from scipy.sparse import csr_matrix
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

reload(sys)
sys.setdefaultencoding("utf8")

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format=('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d :'
            '%(message)s'), datefmt='%a, %d %b %Y %H:%M:%S')


class TextClassifier(object):
    LABEL_TO_INDEX = {'auto':0, 'business':1, 'sports':2}
    INDEX_TO_LABEL = {0:'auto', 1:'business', 2:'sports'}

    def __init__(self, dict_file=None, model_file=None):
        if dict_file:
            self.dictionary = Dictionary.load_from_text(dict_file)
        else:
            self.dictionary = Dictionary()

        if model_file:
            self.model = joblib.load(model_file)
        else:
            self.model = None

    def expand_sent_terms(self, sent, ngrams=[2]):
        expd_sent = list(sent)
        ngram_terms = self._get_ngram_terms(sent, ngrams)
        expd_sent.extend(ngram_terms)

        return expd_sent

    def sentence_to_bow(self, sent):
        if self.dictionary:
            return self.dictionary.doc2bow(sent)
        else:
            return None

    def bow_to_feature_vec(self, bow_corpus):
        data = []
        rows = []
        cols = []
        line_count = 0
        for bow_sent in bow_corpus:
            for elem in bow_sent:
                rows.append(line_count)
                cols.append(elem[0])
                data.append(elem[1])
            line_count += 1

        return csr_matrix(
            (data, (rows,cols)), shape=(line_count, len(self.dictionary)))

    def load_text(self, data_file, train=False):
        term_corpus = []
        labels = []
        with open(data_file) as fin:
            for line in fin:
                parts = line.strip().decode('utf8').split('\t')
                if len(parts) < 2:
                    continue

                label = parts[0]
                sent = parts[1].split()

                # Expand sentence with more features.
                sent = self.expand_sent_terms(sent, [2])

                # Save sentences and labels.
                term_corpus.append(sent)
                labels.append(self.LABEL_TO_INDEX[label])

                # Update dictionary.
                if train:
                    self.dictionary.add_documents([sent])

        if train:
            # Compacitify dictionary.
            self.dictionary.filter_extremes(no_below=5,
                                            no_above=0.6,
                                            keep_n=None)
            self.dictionary.compactify()

        # Change text format corpus to bow format.
        bow_corpus = []
        for sent in term_corpus:
            sent_bow = self.dictionary.doc2bow(sent)
            bow_corpus.append(sent_bow)

        return bow_corpus, labels

    def _get_ngram_terms(self, words, ngrams):
        terms = []
        for i in range(1, len(words)):
            # Bigram terms.
            if 2 in ngrams and (i - 1) >= 0:
                terms.append('%s_%s' % (words[i - 1], words[i]))
            # Trigram terms.
            if 3 in ngrams and (i - 2) >= 0:
                terms.append(
                    '%s_%s_%s' % (words[i - 2], words[i - 1], words[i]))

        return terms

    def dump_dict(self, dict_file):
        self.dictionary.save_as_text(dict_file)

    def dump_model(self, model_file):
        if self.model:
            joblib.dump(self.model, model_file)

    def train(self, x_list, y_list, model='lr'):
        X_train, X_test, y_train, y_test = train_test_split(x_list,
                                                            y_list,
                                                            test_size=0.3)
        if model == 'lr':
            self.model = LogisticRegression(C=1.0,
                                            multi_class='multinomial',
                                            penalty='l2',
                                            solver='sag',
                                            tol=0.1)
        else:
            logging.error('Unknown model name!')
            return

        self.model.fit(X_train, y_train)
        score = self.model.score(X_train, y_train)
        print("Evaluation on train set : %.4f" % score)
        score = self.model.score(X_test, y_test)
        print("Evaluation on test set : %.4f" % score)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def eval(self, X, y):
        score = self.model.score(X, y)
        print("Evaluation on validation set : %.4f" % score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, help='corpus location')
    parser.add_argument('--test_file', type=str, help='corpus location')
    parser.add_argument('--dict_file', type=str, help='dict location')
    parser.add_argument('--model_file', type=str, help='model location')
    parser.add_argument('--ngram', type=int, help='model location')
    args = parser.parse_args()

    if args.train_file:
        # Train.
        clf = TextClassifier()
        sents, labels = clf.load_text(args.train_file, True)
        X_train = clf.bow_to_feature_vec(sents)
        clf.train(X_train, labels)

        # Save model and dictionary.
        if args.model_file:
            clf.dump_model(args.model_file)
        if args.dict_file:
            clf.dump_dict(args.dict_file)
    elif args.dict_file and args.model_file:
        # Init classifier from dictinary and model file.
        clf = TextClassifier(args.dict_file, args.model_file)
    else:
        logging.error('Wrong usage!')
        sys.exit(1)

    # Evaluate on test file.
    if args.test_file:
        sents, labels = clf.load_text(args.test_file)
        X_test = clf.bow_to_feature_vec(sents)
        clf.eval(X_test, labels)

    logging.info('Finished!')

