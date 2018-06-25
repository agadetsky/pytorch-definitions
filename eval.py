from source.datasets import LanguageModelingDataset, LanguageModelingCollate
from source.datasets import DefinitionModelingDataset, DefinitionModelingCollate
from source.datasets import Vocabulary
from source.model import DefinitionModelingModel
from source.constants import BOS
from source.pipeline import test
from source.pipeline import generate
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import torch

parser = argparse.ArgumentParser(description='Script to evaluate model')
parser.add_argument(
    "--params", type=str, required=True,
    help="path to saved model params"
)
parser.add_argument(
    "--ckpt", type=str, required=True,
    help="path to saved model weights"
)
parser.add_argument(
    "--datasplit", type=str, required=True,
    help="train, val or test set to evaluate on"
)
parser.add_argument(
    "--type", type=str, required=True,
    help="compute ppl or bleu"
)
parser.add_argument(
    "--wordlist", type=str, required=False,
    help="word list to evaluate on (by default all data will be used)"
)
# params for BLEU
parser.add_argument(
    "--tau", type=float, required=False,
    help="temperature to use in sampling"
)
parser.add_argument(
    "--n", type=int, required=False,
    help="number of samples to generate"
)
parser.add_argument(
    "--length", type=int, required=False,
    help="maximum length of generated samples"
)
args = parser.parse_args()
assert args.datasplit in ["train", "val", "test"], ("--datasplit must be "
                                                    "train, val or test")
assert args.type in ["ppl", "bleu"], ("--type must be ppl or bleu")

with open(args.params, "r") as infile:
    model_params = json.load(infile)

logfile = open(model_params["exp_dir"] + "eval_log", "a")
#import sys
#logfile = sys.stdout

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DefinitionModelingModel(model_params).to(device)
model.load_state_dict(torch.load(args.ckpt)["state_dict"])

if model.params["pretrain"]:
    assert args.type == "ppl", "if --pretrain True => evaluate only ppl mode"
    if args.datasplit == "train":
        dataset = LanguageModelingDataset(
            file=model.params["train_lm"],
            vocab_path=model.params["voc"],
            bptt=model.params["bptt"],
        )
    elif args.datasplit == "val":
        dataset = LanguageModelingDataset(
            file=model.params["eval_lm"],
            vocab_path=model.params["voc"],
            bptt=model.params["bptt"],
        )
    elif args.datasplit == "test":
        dataset = LanguageModelingDataset(
            file=model.params["test_lm"],
            vocab_path=model.params["voc"],
            bptt=model.params["bptt"],
        )
    dataloader = DataLoader(
        dataset, batch_size=model.params["batch_size"],
        collate_fn=LanguageModelingCollate
    )
else:
    if args.datasplit == "train":
        dataset = DefinitionModelingDataset(
            file=model.params["train_defs"],
            vocab_path=model.params["voc"],
            input_vectors_path=model.params["input_train"],
            input_adaptive_vectors_path=model.params["input_adaptive_train"],
            context_vocab_path=model.params["context_voc"],
            ch_vocab_path=model.params["ch_voc"],
            use_seed=model.params["use_seed"],
            wordlist_path=args.wordlist
        )
    elif args.datasplit == "val":
        dataset = DefinitionModelingDataset(
            file=model.params["eval_defs"],
            vocab_path=model.params["voc"],
            input_vectors_path=model.params["input_eval"],
            input_adaptive_vectors_path=model.params["input_adaptive_eval"],
            context_vocab_path=model.params["context_voc"],
            ch_vocab_path=model.params["ch_voc"],
            use_seed=model.params["use_seed"],
            wordlist_path=args.wordlist
        )
    elif args.datasplit == "test":
        dataset = DefinitionModelingDataset(
            file=model.params["test_defs"],
            vocab_path=model.params["voc"],
            input_vectors_path=model.params["input_test"],
            input_adaptive_vectors_path=model.params["input_adaptive_test"],
            context_vocab_path=model.params["context_voc"],
            ch_vocab_path=model.params["ch_voc"],
            use_seed=model.params["use_seed"],
            wordlist_path=args.wordlist
        )
    dataloader = DataLoader(
        dataset,
        batch_size=1 if args.type == "bleu" else model.params["batch_size"],
        collate_fn=DefinitionModelingCollate
    )
if args.type == "ppl":
    eval_ppl = test(dataloader, model, device, logfile)
else:
    assert args.tau is not None, "--tau is required if --type bleu"
    assert args.n is not None, "--n is required if --type bleu"
    assert args.length is not None, "--length is required if --type bleu"
    defsave = open(
        model.params["exp_dir"] + "generated_" +
        args.datasplit + "_tau=" +
        str(args.tau) + "_n=" + str(args.n) +
        "_length=" + str(args.length) + ".txt",
        "w"
    )
    refsave = open(
        model.params["exp_dir"] + "refs_" + args.datasplit + ".txt",
        "w"
    )
    #defsave = sys.stdout
    voc = Vocabulary()
    voc.load(model.params["voc"])
    to_input = {
        "model": model,
        "voc": voc,
        "tau": args.tau,
        "n": args.n,
        "length": args.length,
        "device": device,
    }
    if model.is_attn:
        context_voc = Vocabulary()
        context_voc.load(model.params["context_voc"])
        to_input["context_voc"] = context_voc
    if model.params["use_ch"]:
        ch_voc = Vocabulary()
        ch_voc.load(model.params["ch_voc"])
        to_input["ch_voc"] = ch_voc
    for i in tqdm(range(len(dataset)), file=logfile):
        if model.is_w2v:
            to_input["input"] = torch.from_numpy(dataset.input_vectors[i])
        if model.is_ada:
            to_input["input"] = torch.from_numpy(
                dataset.input_adaptive_vectors[i]
            )
        if model.is_attn:
            to_input["word"] = dataset.data[i][0][0]
            to_input["context"] = " ".join(dataset.data[i][2])
        if model.params["use_ch"]:
            to_input["CH_word"] = dataset.data[i][0][0]
        if model.params["use_seed"]:
            to_input["prefix"] = dataset.data[i][0][0]
        else:
            to_input["prefix"] = BOS
        defsave.write(
            "Word: {0}\nContext: {1}\n".format(
                dataset.data[i][0][0],
                " ".join(dataset.data[i][2])
            )
        )
        defsave.write(generate(**to_input) + "\n")
        refsave.write(
            "Word: {0}\nContext: {1}\nDefinition: {2}\n".format(
                dataset.data[i][0][0],
                " ".join(dataset.data[i][2]),
                " ".join(dataset.data[i][1])
            )
        )
        defsave.flush()
        logfile.flush()
        refsave.flush()
