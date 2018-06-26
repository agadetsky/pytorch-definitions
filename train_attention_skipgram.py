from source.attention_skipgram import AttentionSkipGram
from source.utils import MultipleOptimizer
import argparse
import numpy as np
import os.path
from tqdm import tqdm
from collections import Counter
import json
from source.datasets import Vocabulary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain

parser = argparse.ArgumentParser(
    description='Script to train a AttentionSkipGram model'
)
parser.add_argument(
    '--data', type=str, required=False,
    help="path to data"
)
parser.add_argument(
    '--context_voc', type=str, required=True,
    help=("path to context voc for DefinitionModelingModel is necessary to "
          "save pretrained attention module, particulary embedding matrix")
)
parser.add_argument(
    '--prepared', dest='prepared', action="store_true",
    help='whether to prepare data or use already prepared'
)
parser.add_argument(
    "--window", type=int, required=True,
    help="window for AttentionSkipGram model"
)
parser.add_argument(
    "--random_seed", type=int, required=True,
    help="random seed for training"
)
parser.add_argument(
    "--sparse", dest="sparse", action="store_true",
    help="whether to use sparse embeddings or not"
)
parser.add_argument(
    "--vec_dim", type=int, required=True,
    help="vector dim to train"
)
parser.add_argument(
    "--attn_hid", type=int, required=True,
    help="hidden size in attention module"
)
parser.add_argument(
    "--attn_dropout", type=float, required=True,
    help="dropout prob in attention module"
)
parser.add_argument(
    "--lr", type=float, required=True,
    help="initial lr to use"
)
parser.add_argument(
    "--batch_size", type=int, required=True,
    help="batch size to use"
)
parser.add_argument(
    "--num_epochs", type=int, required=True,
    help="number of epochs to train"
)
parser.add_argument(
    "--exp_dir", type=str, required=True,
    help="where to save weights, prepared data and logs"
)
args = vars(parser.parse_args())
logfile = open(args["exp_dir"] + "training_log", "a")

context_voc = Vocabulary()
context_voc.load(args["context_voc"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args["random_seed"])
torch.manual_seed(args["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(args["random_seed"])

if args["prepared"]:
    assert os.path.isfile(args["exp_dir"] + "data.npz"), ("prepared data "
                                                          "does not exist")
    assert os.path.isfile(args["exp_dir"] + "voc.json"), ("prepared voc "
                                                          "does not exist")

    tqdm.write("Loading data!", file=logfile)
    logfile.flush()

    data = np.load(args["exp_dir"] + "data.npz")
    words_idx = data['words_idx']
    cnt_idx = data['cnt_idx']
    freqs = data['freqs']

    with open(args["exp_dir"] + "voc.json", 'r') as f:
        voc = json.load(f)

    word2id = voc['word2id']
    id2word = voc['id2word']

else:
    assert args["data"] is not None, "--prepared False, provide --data"

    tqdm.write("Preparing data!", file=logfile)
    logfile.flush()

    with open(args["data"], 'r') as f:
        data = f.read()

    data = data.lower().split()
    counter = Counter(data)
    word2id = {}
    id2word = {}
    i = 0
    words = []
    counts = []
    for w, c in counter.most_common():
        words.append(w)
        counts.append(c)
        word2id[words[-1]] = i
        id2word[i] = w
        i += 1

    freqs = np.array(counts)
    freqs = freqs / freqs.sum()
    freqs = np.sqrt(freqs)
    freqs = freqs / freqs.sum()
    data = list(map(lambda w: word2id[w], data))

    words_idx = np.zeros(len(data) - 2 * args["window"], dtype=np.int)
    cnt_idx = np.zeros(
        (len(data) - 2 * args["window"], 2 * args["window"]), dtype=np.int
    )

    for i in tqdm(range(args["window"], len(data) - args["window"]), file=logfile):
        words_idx[i - args["window"]] = data[i]
        cnt_idx[i - args["window"]] = np.array(
            data[i - args["window"]:i] + data[i + 1:i + args["window"] + 1]
        )

    np.savez(
        args["exp_dir"] + "data",
        words_idx=words_idx,
        cnt_idx=cnt_idx,
        freqs=freqs
    )
    with open(args["exp_dir"] + "voc.json", 'w') as f:
        json.dump({'word2id': word2id, 'id2word': id2word}, f)

    tqdm.write("Data prepared and saved!", file=logfile)
    logfile.flush()


def generate_neg(batch_size, negative=10):
    return np.random.choice(freqs.size, size=(batch_size, negative), p=freqs)


def generate_batch(batch_size=128):
    shuffle = np.random.permutation(words_idx.shape[0])
    words_idx_shuffled = words_idx[shuffle]
    cnt_idx_shuffled = cnt_idx[shuffle]
    for i in tqdm(range(0, words_idx.shape[0], batch_size), file=logfile):
        start = i
        end = min(i + batch_size, words_idx.shape[0])
        words = words_idx_shuffled[start:end]
        context = cnt_idx_shuffled[start:end]
        neg = generate_neg(end - start)

        context = torch.from_numpy(context).to(device)
        words = torch.from_numpy(words).to(device)
        neg = torch.from_numpy(neg).to(device)

        yield words, context, neg

    del words_idx_shuffled
    del cnt_idx_shuffled

tqdm.write("Initialising model!", file=logfile)
logfile.flush()

model = AttentionSkipGram(
    n_attn_tokens=len(word2id),
    n_attn_embsize=args["vec_dim"],
    n_attn_hid=args["attn_hid"],
    attn_dropout=args["attn_dropout"],
    sparse=args["sparse"]
).to(device)


if args["sparse"]:
    optimizer = MultipleOptimizer(
        optim.SparseAdam(chain(
            model.emb0_lookup.embs.parameters(),
            model.emb1_lookup.parameters()
        ), lr=args["lr"]),
        optim.Adam(chain(
            model.emb0_lookup.ann.parameters(),
            model.emb0_lookup.a_linear.parameters()
        ), lr=args["lr"])
    )
else:
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])

tqdm.write("Start training!", file=logfile)
logfile.flush()

model.train()

for _ in range(args["num_epochs"]):
    for w, c, n in generate_batch(batch_size=args["batch_size"]):
        optimizer.zero_grad()
        loss = model(w, c, n)
        loss.backward()
        optimizer.step()


tqdm.write("Training ended! Saving model!", file=logfile)
logfile.flush()

state_dict = model.emb0_lookup.state_dict()
initrange = 0.5 / args["vec_dim"]
embs_weights = np.random.uniform(
    low=-initrange,
    high=initrange,
    size=(len(context_voc.tok2id), args["vec_dim"]),
).astype(np.float32)
for word in context_voc.tok2id.keys():
    if word in word2id:
        new_id = context_voc.tok2id[word]
        old_id = word2id[word]
        embs_weights[new_id] = state_dict["embs.weight"][old_id].cpu().numpy()

state_dict["embs.weight"] = torch.from_numpy(embs_weights).to(device)

torch.save(
    {"state_dict": state_dict},
    args["exp_dir"] + "weights.pth"
)

with open(args["exp_dir"] + "params.json", "w") as outfile:
    json.dump(args, outfile, indent=4)
