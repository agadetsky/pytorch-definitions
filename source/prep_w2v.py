import argparse
from gensim.models import KeyedVectors
import numpy as np
import json

parser = argparse.ArgumentParser(
    description='Prepare word vectors for Input conditioning'
)

parser.add_argument(
    '--defs', type=str, required=True, nargs="+",
    help='location of json file with definitions.'
)

parser.add_argument(
    '--save', type=str, required=True, nargs="+",
    help='where to save files'
)

parser.add_argument(
    "--w2v", type=str, required=True,
    help="location of binary w2v file"
)
args = parser.parse_args()

if len(args.defs) != len(args.save):
    parser.error("Number of defs files must match number of save locations")

word_vectors = KeyedVectors.load_word2vec_format(args.w2v, binary=True)
for i in range(len(args.defs)):
    vectors = []
    with open(args.defs[i], "r") as infile:
        definitions = json.load(infile)
    for elem in definitions:
        if elem[0][0] in word_vectors:
            vectors.append(word_vectors[elem[0][0]])
        else:
            vectors.append(np.zeros(word_vectors.vector_size))
    vectors = np.array(vectors)
    np.save(args.save[i], vectors)
