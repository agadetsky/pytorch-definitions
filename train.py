from source.datasets import LanguageModelingDataset, LanguageModelingCollate
from source.datasets import DefinitionModelingDataset, DefinitionModelingCollate
from source.model import DefinitionModelingModel
from source.pipeline import train_epoch, test
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
import json
import numpy as np

# Read all arguments and prepare all stuff for training

parser = argparse.ArgumentParser(description='Script to train a model')
# Type of training
parser.add_argument(
    '--pretrain', dest='pretrain', action="store_true",
    help='whether to pretrain model on LM dataset or train on definitions'
)
# Common data arguments
parser.add_argument(
    "--voc", type=str, required=True, help="location of vocabulary file"
)
# Definitions data arguments
parser.add_argument(
    '--train_defs', type=str, required=False,
    help="location of json file with train definitions."
)
parser.add_argument(
    '--eval_defs', type=str, required=False,
    help="location of json file with eval definitions."
)
parser.add_argument(
    '--test_defs', type=str, required=False,
    help="location of json file with test definitions"
)
parser.add_argument(
    '--input_train', type=str, required=False,
    help="location of train vectors for Input conditioning"
)
parser.add_argument(
    '--input_eval', type=str, required=False,
    help="location of eval vectors for Input conditioning"
)
parser.add_argument(
    '--input_test', type=str, required=False,
    help="location of test vectors for Input conditioning"
)
parser.add_argument(
    '--input_adaptive_train', type=str, required=False,
    help="location of train vectors for InputAdaptive conditioning"
)
parser.add_argument(
    '--input_adaptive_eval', type=str, required=False,
    help="location of eval vectors for InputAdaptive conditioning"
)
parser.add_argument(
    '--input_adaptive_test', type=str, required=False,
    help="location test vectors for InputAdaptive conditioning"
)
parser.add_argument(
    '--context_voc', type=str, required=False,
    help="location of context vocabulary file"
)
parser.add_argument(
    '--ch_voc', type=str, required=False,
    help="location of CH vocabulary file"
)
# LM data arguments
parser.add_argument(
    '--train_lm', type=str, required=False,
    help="location of txt file train LM data"
)
parser.add_argument(
    '--eval_lm', type=str, required=False,
    help="location of txt file eval LM data"
)
parser.add_argument(
    '--test_lm', type=str, required=False,
    help="location of txt file test LM data"
)
parser.add_argument(
    '--bptt', type=int, required=False,
    help="sequence length for BackPropThroughTime in LM pretraining"
)
# Model parameters arguments
parser.add_argument(
    '--nx', type=int, required=True,
    help="size of embeddings"
)
parser.add_argument(
    '--nlayers', type=int, required=True,
    help="number of LSTM layers"
)
parser.add_argument(
    '--nhid', type=int, required=True,
    help="size of hidden states"
)
parser.add_argument(
    '--rnn_dropout', type=float, required=True,
    help="probability of RNN dropout"
)
parser.add_argument(
    '--use_seed', dest="use_seed", action="store_true",
    help="whether to use Seed conditioning or not"
)
parser.add_argument(
    '--use_input', dest="use_input", action="store_true",
    help="whether to use Input conditioning or not"
)
parser.add_argument(
    '--use_input_adaptive', dest="use_input_adaptive", action="store_true",
    help="whether to use InputAdaptive conditioning or not"
)
parser.add_argument(
    '--use_input_attention', dest="use_input_attention",
    action="store_true",
    help="whether to use InputAttention conditioning or not"
)
parser.add_argument(
    '--n_attn_embsize', type=int, required=False,
    help="size of InputAttention embeddings"
)
parser.add_argument(
    '--n_attn_hid', type=int, required=False,
    help="size of InputAttention linear layer"
)
parser.add_argument(
    '--attn_dropout', type=float, required=False,
    help="probability of InputAttention dropout"
)
parser.add_argument(
    '--attn_sparse', dest="attn_sparse", action="store_true",
    help="whether to use sparse embeddings in InputAttention or not"
)
parser.add_argument(
    '--use_ch', dest="use_ch", action="store_true",
    help="whether to use CH conditioning or not"
)
parser.add_argument(
    '--ch_emb_size', type=int, required=False,
    help="size of embeddings in CH conditioning"
)
parser.add_argument(
    '--ch_feature_maps', type=int, required=False, nargs="+",
    help="list of feature map sizes in CH conditioning"
)
parser.add_argument(
    '--ch_kernel_sizes', type=int, required=False, nargs="+",
    help="list of kernel sizes in CH conditioning"
)
parser.add_argument(
    '--use_hidden', dest="use_hidden", action="store_true",
    help="whether to use Hidden conditioning or not"
)
parser.add_argument(
    '--use_hidden_adaptive', dest="use_hidden_adaptive",
    action="store_true",
    help="whether to use HiddenAdaptive conditioning or not"
)
parser.add_argument(
    '--use_hidden_attention', dest="use_hidden_attention",
    action="store_true",
    help="whether to use HiddenAttention conditioning or not"
)
parser.add_argument(
    '--use_gated', dest="use_gated", action="store_true",
    help="whether to use Gated conditioning or not"
)
parser.add_argument(
    '--use_gated_adaptive', dest="use_gated_adaptive", action="store_true",
    help="whether to use GatedAdaptive conditioning or not"
)
parser.add_argument(
    '--use_gated_attention', dest="use_gated_attention", action="store_true",
    help="whether to use GatedAttention conditioning or not"
)
# Training arguments
parser.add_argument(
    '--lr', type=float, required=True,
    help="initial lr"
)
parser.add_argument(
    "--decay_factor", type=float, required=True,
    help="factor to decay lr"
)
parser.add_argument(
    '--decay_patience', type=int, required=True,
    help="after number of patience epochs - decay lr"
)
parser.add_argument(
    '--num_epochs', type=int, required=True,
    help="number of epochs to train"
)
parser.add_argument(
    '--batch_size', type=int, required=True,
    help="batch size"
)
parser.add_argument(
    "--clip", type=float, required=True,
    help="value to clip norm of gradients to"
)
parser.add_argument(
    "--random_seed", type=int, required=True,
    help="random seed"
)
# Utility arguments
parser.add_argument(
    "--exp_dir", type=str, required=True,
    help="where to save all stuff about training"
)
parser.add_argument(
    "--w2v_weights", type=str, required=False,
    help="path to pretrained embeddings to init"
)
parser.add_argument(
    "--fix_embeddings", dest="fix_embeddings", action="store_true",
    help="whether to update embedding matrix or not"
)
parser.add_argument(
    "--fix_attn_embeddings", dest="fix_attn_embeddings", action="store_true",
    help="whether to update attention embedding matrix or not"
)
parser.add_argument(
    "--lm_ckpt", type=str, required=False,
    help="path to pretrained language model weights"
)
parser.add_argument(
    "--attn_ckpt", type=str, required=False,
    help="path to pretrained Attention module"
)
# read args
args = vars(parser.parse_args())

logfile = open(args["exp_dir"] + "training_log", "a")
#import sys
#logfile = sys.stdout

if args["pretrain"]:
    assert args["train_lm"] is not None, "--train_lm is required if --pretrain"
    assert args["eval_lm"] is not None, "--eval_lm is required if --pretrain"
    assert args["test_lm"] is not None, "--test_lm is required if --pretrain"
    assert args["bptt"] is not None, "--bptt is required if --pretrain"

    train_dataset = LanguageModelingDataset(
        file=args["train_lm"],
        vocab_path=args["voc"],
        bptt=args["bptt"],
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args["batch_size"],
        collate_fn=LanguageModelingCollate
    )
    eval_dataset = LanguageModelingDataset(
        file=args["eval_lm"],
        vocab_path=args["voc"],
        bptt=args["bptt"],
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args["batch_size"],
        collate_fn=LanguageModelingCollate
    )
else:
    assert args["train_defs"] is not None, ("--pretrain is False,"
                                            " --train_defs is required")
    assert args["eval_defs"] is not None, ("--pretrain is False,"
                                           " --eval_defs is required")
    assert args["test_defs"] is not None, ("--pretrain is False,"
                                           " --test_defs is required")

    train_dataset = DefinitionModelingDataset(
        file=args["train_defs"],
        vocab_path=args["voc"],
        input_vectors_path=args["input_train"],
        input_adaptive_vectors_path=args["input_adaptive_train"],
        context_vocab_path=args["context_voc"],
        ch_vocab_path=args["ch_voc"],
        use_seed=args["use_seed"]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        collate_fn=DefinitionModelingCollate
    )
    eval_dataset = DefinitionModelingDataset(
        file=args["eval_defs"],
        vocab_path=args["voc"],
        input_vectors_path=args["input_eval"],
        input_adaptive_vectors_path=args["input_adaptive_eval"],
        context_vocab_path=args["context_voc"],
        ch_vocab_path=args["ch_voc"],
        use_seed=args["use_seed"]
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args["batch_size"],
        collate_fn=DefinitionModelingCollate
    )

    if args["use_input"] or args["use_hidden"] or args["use_gated"]:
        assert args["input_train"] is not None, ("--use_input or "
                                                 "--use_hidden or "
                                                 "--use_gated is used "
                                                 "--input_train is required")
        assert args["input_eval"] is not None, ("--use_input or "
                                                "--use_hidden or "
                                                "--use_gated is used "
                                                "--input_eval is required")
        assert args["input_test"] is not None, ("--use_input or "
                                                "--use_hidden or "
                                                "--use_gated is used "
                                                "--input_test is required")
        args["input_dim"] = train_dataset.input_vectors.shape[1]

    if args["use_input_adaptive"] or args["use_hidden_adaptive"] or args["use_gated_adaptive"]:
        assert args["input_adaptive_train"] is not None, ("--use_input_adaptive or "
                                                          "--use_hidden_adaptive or "
                                                          "--use_gated_adaptive is used "
                                                          "--input_adaptive_train is required")
        assert args["input_adaptive_eval"] is not None, ("--use_input_adaptive or "
                                                         "--use_hidden_adaptive or "
                                                         "--use_gated_adaptive is used "
                                                         "--input_adaptive_eval is required")
        assert args["input_adaptive_test"] is not None, ("--use_input_adaptive or "
                                                         "--use_hidden_adaptive or "
                                                         "--use_gated_adaptive is used "
                                                         "--input_adaptive_test is required")
        args["input_adaptive_dim"] = train_dataset.input_adaptive_vectors.shape[1]

    if args["use_input_attention"] or args["use_hidden_attention"] or args["use_gated_attention"]:
        assert args["context_voc"] is not None, ("--use_input_attention or "
                                                 "--use_hidden_attention or "
                                                 "--use_gated_attention is used "
                                                 "--context_voc is required")
        assert args["n_attn_embsize"] is not None, ("--use_input_attention or "
                                                    "--use_hidden_attention or "
                                                    "--use_gated_attention is used "
                                                    "--n_attn_embsize is required")
        assert args["n_attn_hid"] is not None, ("--use_input_attention or "
                                                "--use_hidden_attention or "
                                                "--use_gated_attention is used "
                                                "--n_attn_hid is required")
        assert args["attn_dropout"] is not None, ("--use_input_attention or "
                                                  "--use_hidden_attention or "
                                                  "--use_gated_attention is used "
                                                  "--attn_dropout is required")

        args["n_attn_tokens"] = len(train_dataset.context_voc.tok2id)

    if args["use_ch"]:
        assert args["ch_voc"] is not None, ("--ch_voc is required "
                                            "if --use_ch")
        assert args["ch_emb_size"] is not None, ("--ch_emb_size is required "
                                                 "if --use_ch")
        assert args["ch_feature_maps"] is not None, ("--ch_feature_maps is "
                                                     "required if --use_ch")
        assert args["ch_kernel_sizes"] is not None, ("--ch_kernel_sizes is "
                                                     "required if --use_ch")

        args["n_ch_tokens"] = len(train_dataset.ch_voc.tok2id)
        args["ch_maxlen"] = train_dataset.ch_voc.tok_maxlen + 2


args["ntokens"] = len(train_dataset.voc.tok2id)

np.random.seed(args["random_seed"])
torch.manual_seed(args["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(args["random_seed"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DefinitionModelingModel(args).to(device)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args["lr"]
    ),
    factor=args["decay_factor"],
    patience=args["decay_patience"]
)

best_ppl = float("inf")
for epoch in tqdm(range(args["num_epochs"]), file=logfile):
    train_epoch(
        train_dataloader,
        model,
        scheduler.optimizer,
        device,
        args["clip"],
        logfile
    )
    eval_ppl = test(eval_dataloader, model, device, logfile)
    if eval_ppl < best_ppl:
        best_ppl = eval_ppl
        torch.save(
            {"state_dict": model.state_dict()},
            args["exp_dir"] + "weights.pth"
        )
    scheduler.step(metrics=eval_ppl)

with open(args["exp_dir"] + "params.json", "w") as outfile:
    json.dump(args, outfile, indent=4)
