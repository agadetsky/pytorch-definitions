import json
import subprocess
import os
import numpy as np
from gensim.models import KeyedVectors
import random


def prepare_ada_vectors_from_python(file, julia_script, ada_binary_path):
    """
    file - path to file with words and contexts on each line separated by \t,
    words and punctuation marks in contexts are separated by spaces
    julia_script - path to prep_ada.jl script
    ada_binary_path - path to ada binary file
    """
    data = open(file, "r").readlines()
    tmp = []
    for i in range(len(data)):
        word, context = data[i].split('\t')
        context = context.strip().split()
        tmp.append([[word], [], context])
    tmp_name = "./tmp" + str(random.randint(1, 999999)) + ".txt"
    tmp_script_name = "./tmp_script" + str(random.randint(1, 999999)) + ".sh"
    tmp_vecs_name = "./tmp_vecs" + str(random.randint(1, 999999))
    with open(tmp_name, "w") as outfile:
        json.dump(tmp, outfile, indent=4)
    with open(tmp_script_name, "w") as outfile:
        outfile.write(
            "julia " + julia_script + " --defs " + tmp_name +
            " --save " + tmp_vecs_name +
            " --ada " + ada_binary_path
        )
    subprocess.call(["/bin/bash", "-i", tmp_script_name])
    vecs = np.load(tmp_vecs_name).astype(np.float32)
    os.remove(tmp_name)
    os.remove(tmp_script_name)
    os.remove(tmp_vecs_name)
    return vecs


def prepare_w2v_vectors(file, w2v_binary_path):
    """
    file - path to file with words and contexts on each line separated by \t,
    words and punctuation marks in contexts are separated by spaces
    w2v_binary_path - path to w2v binary
    """
    data = open(file, "r").readlines()
    word_vectors = KeyedVectors.load_word2vec_format(
        w2v_binary_path, binary=True
    )
    vecs = []
    initrange = 0.5 / word_vectors.vector_size
    for i in range(len(data)):
        word, context = data[i].split('\t')
        context = context.strip().split()
        if word in word_vectors:
            vecs.append(word_vectors[word])
        else:
            vecs.append(
                np.random.uniform(
                    low=-initrange,
                    high=initrange,
                    size=word_vectors.vector_size
                )
            )
    return np.array(vecs, dtype=np.float32)


class MultipleOptimizer(object):

    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
