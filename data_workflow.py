import json
import time
from collections import Counter
import constants
import numpy as np
from utils import flatten
import julia
import gensim
from sklearn.metrics.pairwise import cosine_similarity


class Dictionary(object):

    def __init__(self):
        self.w2i = None
        self.i2w = None

    def size(self):
        return len(self.w2i)

    def encode(self, w):
        if w in self.w2i:
            return self.w2i[w]
        else:
            return constants.UNK

    def decode(self, i):
        return self.i2w[i]

    def w_exists(self, w):
        return w in self.w2i

    def i_exists(self, i):
        return i in self.i2w

    def prepare_dict(self, data, topk=100000):

        cnt = Counter(data)
        topk = cnt.most_common(n=topk)

        self.i2w = {
            constants.PAD: constants.PAD_WORD,
            constants.UNK: constants.UNK_WORD,
            constants.BOS: constants.BOS_WORD,
            constants.EOS: constants.EOS_WORD,
        }
        self.w2i = {
            constants.PAD_WORD: constants.PAD,
            constants.UNK_WORD: constants.UNK,
            constants.BOS_WORD: constants.BOS,
            constants.EOS_WORD: constants.EOS,
        }

        i = 4
        for pair in topk:
            self.w2i[pair[0]] = i
            self.i2w[i] = pair[0]
            i += 1

    def load_from_file(self, path):
        with open(path, "r") as infile:
            self.w2i = json.load(infile)

        self.w2i = {k: int(v) for k, v in self.w2i.items()}
        self.i2w = {int(v): k for k, v in self.w2i.items()}

    def save_to_file(self, path):
        with open(path, "w") as outfile:
            json.dump(self.w2i, outfile)


class WikiCorpus(object):

    def __init__(self, path_to_file=None, topk=None,
                 path_to_dict=None, path_to_prepared_data=None):
        self.path_to_file = path_to_file
        self.path_to_dict = path_to_dict
        self.path_to_prepared_data = path_to_prepared_data
        self.data = None
        self.x_batches = None
        self.y_batches = None
        self.dict = None

        if self.path_to_file is not None and topk is not None:
            self._prepare_data_and_dict(topk=topk)
        elif (self.path_to_dict is not None and
              self.path_to_prepared_data is not None):
            self._load_data_and_dict()
        else:
            raise ValueError("You must provide either path_to_file with topk "
                             "or path_to_dict with path_to_prepared_data!")

    def _load_data_and_dict(self):
        print("Loading dictionary...")
        s = time.time()
        self.dict = Dictionary()
        self.dict.load_from_file(self.path_to_dict)
        print("Dictionary was loaded! "
              "Time elapsed {0} s".format(time.time() - s))

        self.data = []
        print("Loading data...")
        s = time.time()
        with open(self.path_to_prepared_data, "r") as infile:
            for line in infile:
                if self.dict.i_exists(int(line)):
                    self.data.append(int(line))
                else:
                    raise ValueError("No such key {0} "
                                     "in provided dict!".format(int(line)))
        print("Data was loaded. Time elapsed: {0} s".format(time.time() - s))

    def save(self, path_for_dict, path_for_data):
        if self.dict is None or self.data is None:
            raise ValueError("Nothing to save! Either data or dict is None!")

        self.dict.save_to_file(path_for_dict)

        with open(path_for_data, "w") as outfile:
            for elem in self.data:
                outfile.write(str(elem) + "\n")

    def _prepare_data_and_dict(self, topk=1000000):
        self.data = []
        print("Loading data...")
        s = time.time()
        with open(self.path_to_file, "r") as infile:
            for line in infile:
                self.data.append(line.strip())
        print("Data was loaded. Time elapsed: {0} s".format(time.time() - s))

        print("Building dictionary...")
        s = time.time()
        self.dict = Dictionary()
        self.dict.prepare_dict(self.data, topk=topk)
        print("Dictionary was built. "
              "Time elapsed: {0} s".format(time.time() - s))

        print("Encoding data...")
        s = time.time()
        for i, word in enumerate(self.data):
            self.data[i] = self.dict.encode(word)
        print("Data was encoded. Time elapsed: {0} s".format(time.time() - s))

    def prepare_batches(self, batch_size=96, seqlen=35):
        print("Prepairing batches...")
        s = time.time()

        x = []
        y = []
        nseq = len(self.data) // seqlen
        nbatches = nseq // batch_size
        if nbatches <= 0:
            raise ValueError("Batch size or maxseqlen is too big!")
        upper_bound = nbatches * batch_size * seqlen

        for i in range(0, upper_bound, seqlen):
            x.append([constants.BOS] + self.data[i:i + seqlen])
            y.append(self.data[i:i + seqlen] + [constants.EOS])

        x = np.array(x)
        y = np.array(y)

        self.x_batches = np.split(x, nbatches)
        self.y_batches = np.split(y, nbatches)
        print("Batches were prepared. "
              "Time elapsed: {0} s".format(time.time() - s))


class Definitions(object):

    def __init__(self, path_to_file=None, path_to_dict=None,
                 topk=100000, cond=None):
        self.path_to_file = path_to_file
        self.path_to_dict = path_to_dict
        self.data = None
        self.x_batches = None
        self.y_batches = None
        self.dict = None

        if self.path_to_file is None:
            raise ValueError("You must provide path_to_file!")

        print("Loading data...")
        s = time.time()
        with open(self.path_to_file, "r") as infile:
            self.data = json.load(infile)
        print("Data was loaded. Time elapsed: {0} s".format(time.time() - s))

        if self.path_to_dict is None:
            self.dict = Dictionary()
            self.dict.prepare_dict(flatten(self.data), topk=topk)
        else:
            self.dict = Dictionary()
            self.dict.load_from_file(self.path_to_dict)

        self.cond = cond

    def prepare_data(self):
        x = []
        y = []
        conds = []

        for i in range(len(self.data)):
            sentence = self.data[i]

            #x_i = [constants.BOS, ]
            x_i = []
            x_i.extend([self.dict.encode(w) for w in sentence])

            y_i = x_i[1:]
            y_i.append(constants.EOS)

            cond_i = self.cond.get_cond_vector(sentence[0], sentence[1:])

            x.append(x_i)
            y.append(y_i)
            conds.append(cond_i)

        return x, y, conds

    def prepare_batch(self, x, y, conds):
        assert len(x) == len(y) == len(conds)

        batch_size = len(x)

        lengths = []
        for x_i in x:
            lengths.append(len(x_i))

        maxlen = max(lengths)

        len_argsort = np.argsort(lengths)[::-1]

        x_batch = np.zeros((batch_size, maxlen), dtype=int)
        y_batch = np.zeros((batch_size, maxlen), dtype=int)

        for i, pos in enumerate(len_argsort):
            x_batch[i, :lengths[pos]] = x[pos]
            y_batch[i, :lengths[pos]] = y[pos]

        return x_batch, y_batch, np.array(conds)[len_argsort]


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
