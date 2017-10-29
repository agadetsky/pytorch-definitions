import json
import time
from collections import Counter
import constants
import numpy as np
import julia
import gensim
from sklearn.metrics.pairwise import cosine_similarity


class Zeros(object):

    def __init__(self, ndim):
        self.ndim = ndim

    def get_cond_vector(self, word, context=None):
        return np.zeros(self.ndim)


class Word2Vec(object):

    def __init__(self, path_to_file):

        self.m = gensim.models.KeyedVectors.load_word2vec_format(
            path_to_file,
            binary=True
        )

        self.i2w = self.m.index2word
        self.w2i = {self.i2w[i]: i for i in range(len(self.i2w))}

        self.ncond = self.m.vector_size

    def get_cond_vector(self, word, context=None):
        if word not in self.w2i:
            return np.zeros(self.ncond)
        return self.m.wv[word]

    def get_word_neighbours(self, word, k=10):
        return self.m.similar_by_word(word=word, topn=k)


class AdaGram(object):

    def __init__(self, path_to_file):
        print("Initialising Julia!")
        self.j = julia.Julia()
        print("Importing AdaGram!")
        ret = self.j.eval('using AdaGram')
        print("Loading Model!")
        ret = self.j.eval('vm, dict = load_model("{0}");'.format(path_to_file))
        print("Prepairing dict!")
        self.w2i = self.j.eval('dict.word2id')
        self.i2w = self.j.eval('dict.id2word')
        self.ncond = self.j.eval('size(vm.In)[1]')

    def get_meaning(self, word, context):
        if word not in self.w2i:
            raise ValueError("No such word in model dictionary!")

        good_context = []
        for c_i in context:
            if c_i in self.w2i:
                good_context.append(c_i)

        probs = self.j.eval(
            'disambiguate(vm, dict, '
            '"{0}", split("{1}"))'.format(word, " ".join(good_context))
        )

        return np.argmax(probs)

    def get_cond_vector(self, word, context):
        if word not in self.w2i:
            return np.zeros(self.ncond)

        arg = self.get_meaning(word, context) + 1

        vec = self.j.eval('vm.In[:, {0},'
                          ' dict.word2id["{1}"]]'.format(arg, word))
        return vec

    def get_priors(self, word):
        if word not in self.w2i:
            raise ValueError("No such word in model dictionary!")

        probs = self.j.eval(
            'expected_pi(vm, dict.word2id["{0}"])'.format(word)
        )
        return probs

    def get_word_neighbours(self, word, meaning, k=10):
        if word not in self.w2i:
            raise ValueError("No such word in model dictionary!")

        ranking = self.j.eval(
            'nearest_neighbors(vm, dict, '
            '"{0}", {1}, {2})'.format(word, meaning + 1, k)
        )

        return ranking

    def get_vec_by_meaning(self, word, meaning):
        if word not in self.w2i:
            raise ValueError("No such word in model dictionary!")

        vec = self.j.eval(
            'vm.In[:, {0}, dict.word2id["{1}"]]'.format(meaning + 1, word)
        )

        return vec
