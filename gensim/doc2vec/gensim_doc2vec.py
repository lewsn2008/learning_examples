#!/usr/bin/env python
#coding=utf8
from __future__ import print_function
import argparse
import re
import sys
import numpy as np

from random import shuffle

from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

reload(sys)
sys.setdefaultencoding("utf8")

DATE_PTN = re.compile(u'(((19)*9\d|(20)*[01]\d)\-?)?((0[1-9]|1[012])\-?)([012]\d|3[01])')
def_labels = ['movie', 'episode', 'enter', 'cartoon', 'game']
label_to_index = {'movie':0, 'episode':1, 'enter':2, 'cartoon':3, 'game':4}
index_to_label = {0:'movie', 1:'episode', 2:'enter', 3:'cartoon', 4:'game'}


class LabeledLineSentence(object):
    def __init__(self):
        self._reset()

    def _reset(self):
        self.labels = []
        self.labelled_sentences = []

    def load(self, train_file):
        self._reset()
        with open(train_file) as fin:
            for line_num, line in enumerate(fin):
                parts = line.strip().decode('utf8').split('\t')
                if len(parts) < 3:
                    continue

                label, keyword = parts[0:2]
                # 将句子中的日期字符串(如“20120813”)归一化为"@date@".
                sent = ['@date@' if DATE_PTN.match(term) else term for term in parts[2:]]

                # 给句子标记tag, 每个句子可以是多个tag, 可以是句子的序号(number), 
                # 或者唯一的标记(如：sent_i), 此处用的是WSD(词义消歧)中的词义为tag,
                # 例如“花千骨”有game和episode两种词义, 对于标记了词义的句子可以打上
                # 不同词义的tag, 例如：game_花千骨, episode_花千骨.
                #sent_tag = 'SENT_%d' % line_num
                keyword_tag = '%s_%s' % (label, keyword)

                # 构造gensim的LabeledSentence表示一个标记了tag的句子.
                labeled_sent = LabeledSentence(sent, [keyword_tag])

                # 保存句子和分类标签.
                self.labelled_sentences.append(labeled_sent)
                self.labels.append(label_to_index[label])


class Doc2VecModel(object):
    def __init__(self):
        self.labelled_corpus = None
        self.sense_tag_vec_dict = {}
        self.doc2vec_dim = 0

    def train(self, args):
        if not args.train_file:
            print('Train file path needed!')
            sys.exit(1)

        # Load corpus from file.
        self.labelled_corpus = LabeledLineSentence()
        self.labelled_corpus.load(args.train_file)

        # Define doc2vec object and build vocabulary.
        labelled_sents = self.labelled_corpus.labelled_sentences
        self.doc2vec = Doc2Vec(min_count=1, size=args.dimension, window=15)
        self.doc2vec.build_vocab(labelled_sents)

        # Train doc2vec.
        for epoch in range(args.epoch):
            print('epoch %d' % epoch)
            shuffle(labelled_sents)
            self.doc2vec.train(labelled_sents)

        self.doc2vec_dim = args.dimension
        self.__save_sense_tag_vecs()
        print('Finished training!')

    def dump(self, args):
        if args.model_file:
            self.doc2vec.save(args.model_file)

    def load(self, args):
        if args.model_file:
            self.doc2vec = Doc2Vec.load(args.model_file)
        self.doc2vec_dim = self.doc2vec.docvecs[0].shape[0]
        self.__save_sense_tag_vecs()

    def __save_sense_tag_vecs(self):
        for i in range(len(self.doc2vec.docvecs)):
            sense_tag = self.doc2vec.docvecs.index_to_doctag(i)
            sense_tag_vec = self.doc2vec.docvecs[i]
            self.sense_tag_vec_dict[sense_tag] = (
                sense_tag_vec / np.linalg.norm(sense_tag_vec))

    def eval_test(self, test_file):
        with open(test_file) as fin:
            correct_num = 0
            for line_num, line in enumerate(fin):
                parts = line.strip().decode('utf8').split('\t')
                if len(parts) < 3:
                    continue

                target_label, keyword = parts[0:2]
                sent = ['@date@' if DATE_PTN.match(term) else term for term in parts[2:]]
                if keyword not in sent:
                    continue

                # Context text embedding similarities {{{.
                sent_vec = self.doc2vec.infer_vector(
                        sent, alpha=0.1, min_alpha=0.0001, steps=5)
                sent_vec = sent_vec / np.linalg.norm(sent_vec)
                sense_tag_vecs = self.get_sense_tag_vecs(parts[1])
                sims = sent_vec.dot(sense_tag_vecs.T)
                sims = (sims + 1) / 2   # normalization
                # }}}.

                pred_label = np.argmax(sims)
                if label_to_index[target_label] == pred_label:
                    correct_num += 1

        print('Eval on test set: precision : %f (%d/%d)' %
            (correct_num/float(line_num), correct_num, line_num))

    def get_sense_tag_vecs(self, word):
        if not isinstance(word, unicode):
            word = word.decode('utf8')

        default_vec = np.zeros(self.doc2vec_dim)
        vecs = []
        for label in def_labels:
            sense_tag = '%s_%s' % (label, word)
            sense_tag_vec = self.sense_tag_vec_dict.get(sense_tag, default_vec)
            vecs.append(sense_tag_vec.tolist())

        return np.array(vecs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, help='train file location')
    parser.add_argument('--test_file', type=str, help='test location')
    parser.add_argument('--dimension', type=int, default=200, help='vector dimension')
    parser.add_argument('--epoch', type=int, default=1, help='number of epoch for training')
    parser.add_argument('--model_file', type=str, help='model location')
    args = parser.parse_args()

    model = Doc2VecModel()
    if args.train_file:
        model.train(args)
        model.dump(args)
    elif args.model_file:
        model.load(args)
    else:
        print('Wrong usage!')
        sys.exit(1)

    if args.test_file:
        model.eval_test(args.test_file)

