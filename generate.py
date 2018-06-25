from source.model import DefinitionModelingModel
from source.pipeline import generate
from source.datasets import Vocabulary
from source.utils import prepare_ada_vectors_from_python, prepare_w2v_vectors
from constants import BOS
import argparse
import torch
import json

parser = argparse.ArgumentParser(description='Script to generate using model')
parser.add_argument(
    "--params", type=str, required=True,
    help="path to saved model params"
)
parser.add_argument(
    "--ckpt", type=str, required=True,
    help="path to saved model weights"
)
parser.add_argument(
    "--tau", type=float, required=True,
    help="temperature to use in sampling"
)
parser.add_argument(
    "--n", type=int, required=True,
    help="number of samples to generate"
)
parser.add_argument(
    "--length", type=int, required=True,
    help="maximum length of generated samples"
)
parser.add_argument(
    "--prefix", type=str, required=False,
    help="prefix to read until generation starts"
)
parser.add_argument(
    "--wordlist", type=str, required=False,
    help="path to word list with words and contexts"
)

args = parser.parse_args()

with open(args.params, "r") as infile:
    model_params = json.load(infile)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DefinitionModelingModel(model_params).to(device)
model.load_state_dict(torch.load(args.ckpt)["state_dict"])
voc = Vocabulary()
voc.load(model_params["voc"])
to_input = {
    "model": model,
    "voc": voc,
    "tau": args.tau,
    "n": args.n,
    "length": args.length,
    "device": device,
}
if model.params["pretrain"]:
    to_input["prefix"] = args.prefix
    print(generate(**to_input))
else:
    assert args.wordlist is not None, ("to generate definitions in --pretrain "
                                       "False mode --wordlist is required")

    with open(args.wordlist, "r") as infile:
        data = infile.readlines()

    if model.is_w2v:
        input_vecs = torch.from_numpy(
            prepare_w2v_vectors(args.wordlist)
        )
    if model.is_ada:
        input_vecs = torch.from_numpy(
            prepare_ada_vectors_from_python(args.wordlist)
        )
    if model.is_attn:
        context_voc = Vocabulary()
        context_voc.load(model.params["context_voc"])
        to_input["context_voc"] = context_voc
    if model.params["use_ch"]:
        ch_voc = Vocabulary()
        ch_voc.load(model.params["ch_voc"])
        to_input["ch_voc"] = ch_voc
    for i in range(len(data)):
        word, context = data[i].split('\t')
        context = context.strip()
        if model.is_w2v or model.is_ada:
            to_input["input"] = input_vecs[i]
        if model.is_attn:
            to_input["word"] = word
            to_input["context"] = context
        if model.params["use_ch"]:
            to_input["CH_word"] = word
        if model.params["use_seed"]:
            to_input["prefix"] = word
        else:
            to_input["prefix"] = BOS
        print(generate(**to_input))
