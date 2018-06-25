import json
import subprocess
import os
import numpy as np
from gensim.models import KeyedVectors
from constants import ADA_BINARY_PATH, JULIA_SCRIPT, W2V_BINARY_PATH
import random


def prepare_ada_vectors_from_python(file):
    """
    file - path to file with words and contexts on each line separated by \t,
    words and punctuation marks in contexts are separated by spaces
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
            "julia " + JULIA_SCRIPT + " --defs " + tmp_name +
            " --save " + tmp_vecs_name +
            " --ada " + ADA_BINARY_PATH
        )
    subprocess.call(["/bin/bash", "-i", tmp_script_name])
    vecs = np.load(tmp_vecs_name).astype(np.float32)
    os.remove(tmp_name)
    os.remove(tmp_script_name)
    os.remove(tmp_vecs_name)
    return vecs


def prepare_w2v_vectors(file):
    """
    file - path to file with words and contexts on each line separated by \t,
    words and punctuation marks in contexts are separated by spaces
    """
    data = open(file, "r").readlines()
    word_vectors = KeyedVectors.load_word2vec_format(
        W2V_BINARY_PATH, binary=True
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
