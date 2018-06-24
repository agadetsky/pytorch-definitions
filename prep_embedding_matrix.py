import argparse
from gensim.models import KeyedVectors
import torch
import numpy as np
from source.datasets import Vocabulary

parser = argparse.ArgumentParser(
    description='Prepare word vectors for embedding layer in the model'
)
parser.add_argument(
    "--voc", type=str, required=True,
    help="location of model vocabulary file"
)
parser.add_argument(
    "--w2v", type=str, required=True,
    help="location of binary w2v file"
)
parser.add_argument(
    "--save", type=str, required=True,
    help="where to save prepaired matrix"
)
args = parser.parse_args()
word_vectors = KeyedVectors.load_word2vec_format(args.w2v, binary=True)
voc = Vocabulary()
voc.load(args.voc)
vecs = []
initrange = 0.5 / word_vectors.vector_size
for key in voc.tok2id.keys():
    if key in word_vectors:
        vecs.append(word_vectors[key])
    else:
        vecs.append(
            np.random.uniform(
                low=-initrange,
                high=initrange,
                size=word_vectors.vector_size)
        )
torch.save(torch.from_numpy(np.array(vecs)).float(), args.save)
