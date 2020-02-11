#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>

"""
F19 11-411/611 NLP Assignment 3 Task 1
N-gram Language Model Implementation Script
Zimeng Qiu Sep 2019

This is a simple implementation of N-gram language model

Write your own implementation in this file!
"""

import argparse
import math
from utils import *


class LanguageModel(object):
    """
    Base class for all language models
    """
    
    def __init__(self, corpus, ngram, min_freq, uniform=False):
        """
        Initialize language model
        :param corpus: input text corpus to build LM on
        :param ngram: number of n-gram, e.g. 1, 2, 3, ...
        :param min_freq: minimum frequency threshold to set a word to UNK placeholder
                         set to 1 to not use this threshold
        :param uniform: boolean flag, set to True to indicate this model is a simple uniform LM
                        otherwise will be an N-gram model
        """
        # write your initialize code below
        
        self.ngram = ngram
        self.min_freq = min_freq
        self.uniform = uniform
        self.corpus = corpus
        self.gram = None
    def build(self):
        """
        Build LM from text corpus
        """
        # Write your own implementation here
        # initialize corpus
        corpus_dict = {}
        for sentence in self.corpus:
            for token in sentence:
                if token in corpus_dict:
                    corpus_dict[token] += 1
                else:
                    corpus_dict[token] = 1

        replace_list = []
        for gram in corpus_dict:
            if corpus_dict[gram] < self.min_freq:
                replace_list.append(gram)

        for i in range(len(self.corpus)):
            for j in range(len(self.corpus[i])):
                if self.corpus[i][j] in replace_list:
                    self.corpus[i][j] = 'UNK'

        if self.ngram == 1:
            if self.uniform == True:
                uniform = {}
                for sentence in self.corpus:
                    for token in sentence:
                        uniform[token] = 1
                self.gram = uniform

            else:
                unigram = {}
                for sentence in self.corpus:
                    for token in sentence:
                        if token in unigram:
                            unigram[token] += 1
                        else:
                            unigram[token] = 1
                self.gram = unigram

        if self.ngram == 2:
            bigram = {}
            for sentence in self.corpus:
                for i in range(len(sentence) - 1):
                    token = sentence[i] + ' ' + sentence[i+1]
                    if token in bigram:
                        bigram[token] += 1
                    else:
                        bigram[token] = 1
            self.gram = bigram

        if self.ngram == 3:
            trigram = {}
            for sentence in self.corpus:
                for i in range(len(sentence) - 2):
                    token = sentence[i] + ' ' + sentence[i+1] + ' ' + sentence[i+2]
                    if token in trigram:
                        trigram[token] += 1
                    else:
                        trigram[token] = 1
            self.gram = trigram
            
        

    def most_common_words(self, k):
        """
        This function will only be called after the language model has been built
        Your return should be sorted in descending order of frequency
        Sort according to ascending alphabet order when multiple words have same frequency
        :return: list[tuple(token, freq)] of top k most common tokens
        """
        # Write your own implementation here
        self.build()
        sorted_list = []
        for key, value in sorted(self.gram.items(), key=lambda item: (1 / item[1], item[0]), reverse=False):
            sorted_list.append((key, value))
        return sorted_list[:k]


def calculate_perplexity(models, coefs, data):
    """
    Calculate perplexity with given model
    :param models: language models
    :param coefs: coefficients
    :param data: test data
    :return: perplexity
    """
    # Write your own implementation here
    models = models
    coefs = coefs
    data = data
    prop = []
    uniform_prop = []
    unigram_prop = []
    bigram_prop = []
    trigram_prop = []
    pp = 0
    unigram_model = None
    bigram_model = None
    N = None
    training_corpus = None
    for model in models:
        if model.ngram == 1:
            if model.uniform == True:
                training_corpus = model.most_common_words(k=None)
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] not in [m for m,n in training_corpus]:
                data[i][j] = 'UNK'
    for i in range(len(models)):
        if models[i].ngram == 1:
            if models[i].uniform == True:
                for sentence in data:
                    for key in sentence:
                        uniform_prop.append(coefs[i] * (1.0 / len(training_corpus)))
            
            if models[i].uniform == False:
                models[i].build()
                unigram_model = models[i].gram
                Nuni = 0
                for _, n in unigram_model.items():
                    Nuni += n
                N = Nuni
                for sentence in data:
                    for key in sentence:
                        unigram_prop.append(coefs[i] * (unigram_model[key] / N))
    
        if models[i].ngram == 2:
            models[i].build()
            bigram_model = models[i].gram
            
            for sentence in data:
                bigram_prop.append(coefs[i])
                for j in range(1, len(sentence)):
                    w = sentence[j-1]
                    bi = sentence[j-1] + ' ' + sentence[j]
                    if bi in bigram_model:
                        p = bigram_model[bi] / unigram_model[w]
                        bigram_prop.append(coefs[i] * p)
                    else:
                        p = 1 / (unigram_model[w] + len(unigram_model))
                        bigram_prop.append(coefs[i] * p)

        if models[i].ngram == 3:
            models[i].build()
            trigram_model = models[i].gram

            for sentence in data:
                trigram_prop.append(coefs[i])
                trigram_prop.append(coefs[i])
                for j in range(2, len(sentence)):
                    w = sentence[j-2]
                    bi = sentence[j-2] + ' ' + sentence[j-1]
                    tri = sentence[j-2] + ' ' + sentence[j-1] + ' ' + sentence[j]
                    if tri in trigram_model:
                        p = (trigram_model[tri] + 1) / bigram_model[bi]
                        trigram_prop.append(coefs[i] * p)
                    else:
                        if bi in bigram_model:
                            p = (bigram_model[bi] + 1) / unigram_model[w]
                            trigram_prop.append(coefs[i] * p)
                        else:
                            p = unigram_model[w] / len(unigram_model)
                            trigram_prop.append(coefs[i] * p)




    n_w = 0    
    for sentence in data:
        for token in sentence:
            n_w += 1
            prop.append(0)
    if len(uniform_prop) != 0:
        for i in range(len(prop)):
            prop[i] += uniform_prop[i]
    
    if len(unigram_prop) != 0:
        for i in range(len(prop)):
            prop[i] += unigram_prop[i]

    if len(bigram_prop) != 0:
        for i in range(len(prop)):
            prop[i] += bigram_prop[i]
    
    if len(trigram_prop) != 0:
        for i in range(len(prop)):
            prop[i] += trigram_prop[i]

    for p in prop:
        pp += math.log2(p * 10000) - math.log2(10000)
    
    pp /= (-n_w)
    pp = math.pow(2, pp)

    return pp
                


                



# Do not modify this function!
def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('N-gram Language Model')
    parser.add_argument('coef_unif', help='coefficient for the uniform model.', type=float)
    parser.add_argument('coef_uni', help='coefficient for the unigram model.', type=float)
    parser.add_argument('coef_bi', help='coefficient for the bigram model.', type=float)
    parser.add_argument('coef_tri', help='coefficient for the trigram model.', type=float)
    parser.add_argument('min_freq', type=int,
                        help='minimum frequency threshold for substitute '
                             'with UNK token, set to 1 for not use this threshold')
    parser.add_argument('testfile', help='test text file.')
    parser.add_argument('trainfile', help='training text file.', nargs='+')
    args = parser.parse_args()
    return args


# Main executable script provided for your convenience
# Not executed on autograder, so do what you want
if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # load and preprocess train and test data
    train = preprocess(load_dataset(args.trainfile))
    test = preprocess(read_file(args.testfile))

    # build language models
    uniform = LanguageModel(train, ngram=1, min_freq=args.min_freq, uniform=True)
    unigram = LanguageModel(train, ngram=1, min_freq=args.min_freq)
    bigram = LanguageModel(train, ngram=2, min_freq=args.min_freq)
    trigram = LanguageModel(train, ngram=3, min_freq=args.min_freq)

    # calculate perplexity on test file
    ppl = calculate_perplexity(
        models=[uniform, unigram, bigram, trigram],
        coefs=[args.coef_unif, args.coef_uni, args.coef_bi, args.coef_tri],
        data=test)

    print("Perplexity: {}".format(ppl))





