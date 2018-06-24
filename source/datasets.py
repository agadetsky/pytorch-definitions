from torch.utils.data import Dataset
import constants
import json
import numpy as np
import math


class Vocabulary:
    """Word/char vocabulary"""

    def __init__(self):
        self.tok2id = {
            constants.PAD: constants.PAD_IDX,
            constants.UNK: constants.UNK_IDX,
            constants.BOS: constants.BOS_IDX,
            constants.EOS: constants.EOS_IDX
        }
        self.id2tok = {
            constants.PAD_IDX: constants.PAD,
            constants.UNK_IDX: constants.UNK,
            constants.BOS_IDX: constants.BOS,
            constants.EOS_IDX: constants.EOS
        }

        # we need this for maxlen of word being definedR in CH conditioning
        self.tok_maxlen = -float("inf")

    def encode(self, tok):
        if tok in self.tok2id:
            return self.tok2id[tok]
        else:
            return constants.UNK_IDX

    def decode(self, idx):
        if idx in self.id2tok:
            return self.id2tok[idx]
        else:
            raise ValueError("No such idx: {0}".format(idx))

    def encode_seq(self, arr):
        ret = []
        for elem in arr:
            ret.append(self.encode(elem))
        return ret

    def decode_seq(self, arr):
        ret = []
        for elem in arr:
            ret.append(self.decode(elem))
        return ret

    def add_token(self, tok):
        if tok not in self.tok2id:
            self.tok2id[tok] = len(self.tok2id)
            self.id2tok[len(self.id2tok)] = tok

    def save(self, path):
        with open(path, "w") as outfile:
            json.dump([self.id2tok, self.tok_maxlen], outfile, indent=4)

    def load(self, path):
        with open(path, "r") as infile:
            self.id2tok, self.tok_maxlen = json.load(infile)
        self.id2tok = {int(k): v for k, v in self.id2tok.items()}
        self.tok2id = {}
        for i in self.id2tok.keys():
            self.tok2id[self.id2tok[i]] = i


def pad(seq, size, value):
    if len(seq) < size:
        seq.extend([value] * (size - len(seq)))
    return seq


class LanguageModelingDataset(Dataset):
    """LanguageModeling dataset."""

    def __init__(self, file, vocab_path, bptt):
        """
        Args:
            file (string): Path to the file
            vocab_path (string): path to word vocab to use
            bptt (int): length of one sentence
        """
        with open(file, "r") as infile:
            self.data = infile.read().lower().split()
        self.voc = Vocabulary()
        self.voc.load(vocab_path)
        self.bptt = bptt

    def __len__(self):
        return math.ceil(len(self.data) / (self.bptt + 1))

    def __getitem__(self, idx):
        i = idx + self.bptt * idx
        sample = {
            "x": self.voc.encode_seq(self.data[i: i + self.bptt]),
            "y": self.voc.encode_seq(self.data[i + 1: i + self.bptt + 1]),
        }
        return sample


def LanguageModelingCollate(batch):
    batch_x = []
    batch_y = []
    maxlen = -float("inf")
    for i in range(len(batch)):
        batch_x.append(batch[i]["x"])
        batch_y.append(batch[i]["y"])
        maxlen = max(maxlen, len(batch[i]["x"]), len(batch[i]["y"]))

    for i in range(len(batch)):
        batch_x[i] = pad(batch_x[i], maxlen, constants.PAD_IDX)
        batch_y[i] = pad(batch_y[i], maxlen, constants.PAD_IDX)

    ret_batch = {
        "x": np.array(batch_x),
        "y": np.array(batch_y),
    }
    return ret_batch


class DefinitionModelingDataset(Dataset):
    """DefinitionModeling dataset."""

    def __init__(self, file, vocab_path, input_vectors_path=None,
                 input_adaptive_vectors_path=None, context_vocab_path=None,
                 ch_vocab_path=None, use_seed=False):
        """
        Args:
            file (string): path to the file
            vocab_path (string): path to word vocab to use
            input_vectors_path (string): path to vectors for Input conditioning
            input_adaptive_vectors_path (string): path to vectors for Input-Adaptive conditioning
            context_vocab_path (string): path to vocab for context words for Input-Attention
            ch_vocab_path (string): path to char vocab for CH conditioning
            use_seed (bool): whether to use Seed conditioning or not
        """
        with open(file, "r") as infile:
            self.data = json.load(infile)
        self.voc = Vocabulary()
        self.voc.load(vocab_path)
        if context_vocab_path is not None:
            self.context_voc = Vocabulary()
            self.context_voc.load(context_vocab_path)
        if ch_vocab_path is not None:
            self.ch_voc = Vocabulary()
            self.ch_voc.load(ch_vocab_path)
        if input_vectors_path is not None:
            self.input_vectors = np.load(input_vectors_path)
        if input_adaptive_vectors_path is not None:
            self.input_adaptive_vectors = np.load(input_adaptive_vectors_path)
        self.use_seed = use_seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            "x": self.voc.encode_seq(self.data[idx][1]),
            "y": self.voc.encode_seq(self.data[idx][1][1:] + [constants.EOS]),
        }
        if hasattr(self, "input_vectors"):
            sample["input"] = self.input_vectors[idx]
        if hasattr(self, "input_adaptive_vectors"):
            sample["input_adaptive"] = self.input_adaptive_vectors[idx]
        if hasattr(self, "context_voc"):
            sample["word"] = self.context_voc.encode(self.data[idx][0][0])
            sample["context"] = self.context_voc.encode_seq(self.data[idx][2])
        if hasattr(self, "ch_voc"):
            sample["CH"] = [constants.BOS_IDX] + \
                self.ch_voc.encode_seq(list(self.data[idx][0][0])) + \
                [constants.EOS_IDX]
            # CH_maxlen: +2 because EOS + BOS
            sample["CH_maxlen"] = self.ch_voc.tok_maxlen + 2
        if self.use_seed:
            sample["y"] = [sample["x"][0]] + sample["y"]
            sample["x"] = self.voc.encode_seq(self.data[idx][0]) + sample["x"]
        return sample


def DefinitionModelingCollate(batch):
    batch_x = []
    batch_y = []
    is_w2v = "input" in batch[0]
    is_ada = "input_adaptive" in batch[0]
    is_attn = "word" in batch[0] and "context" in batch[0]
    is_ch = "CH" in batch[0] and "CH_maxlen" in batch[0]
    if is_w2v:
        batch_input = []
    if is_ada:
        batch_input_adaptive = []
    if is_attn:
        batch_word = []
        batch_context = []
        context_maxlen = -float("inf")
    if is_ch:
        batch_ch = []
        CH_maxlen = batch[0]["CH_maxlen"]

    definition_lengths = []
    for i in range(len(batch)):
        batch_x.append(batch[i]["x"])
        batch_y.append(batch[i]["y"])
        if is_w2v:
            batch_input.append(batch[i]["input"])
        if is_ada:
            batch_input_adaptive.append(batch[i]["input_adaptive"])
        if is_attn:
            batch_word.append(batch[i]["word"])
            batch_context.append(batch[i]["context"])
            context_maxlen = max(context_maxlen, len(batch_context[-1]))
        if is_ch:
            batch_ch.append(batch[i]["CH"])
        definition_lengths.append(len(batch_x[-1]))

    definition_maxlen = max(definition_lengths)

    for i in range(len(batch)):
        batch_x[i] = pad(batch_x[i], definition_maxlen, constants.PAD_IDX)
        batch_y[i] = pad(batch_y[i], definition_maxlen, constants.PAD_IDX)
        if is_attn:
            batch_context[i] = pad(
                batch_context[i], context_maxlen, constants.PAD_IDX
            )
        if is_ch:
            batch_ch[i] = pad(batch_ch[i], CH_maxlen, constants.PAD_IDX)

    order = np.argsort(definition_lengths)[::-1]
    batch_x = np.array(batch_x)[order]
    batch_y = np.array(batch_y)[order]
    ret_batch = {
        "x": batch_x,
        "y": batch_y,
    }
    if is_w2v:
        batch_input = np.array(batch_input, dtype=np.float32)[order]
        ret_batch["input"] = batch_input
    if is_ada:
        batch_input_adaptive = np.array(
            batch_input_adaptive,
            dtype=np.float32
        )[order]
        ret_batch["input_adaptive"] = batch_input_adaptive
    if is_attn:
        batch_word = np.array(batch_word)[order]
        batch_context = np.array(batch_context)[order]
        ret_batch["word"] = batch_word
        ret_batch["context"] = batch_context
    if is_ch:
        batch_ch = np.array(batch_ch)[order]
        ret_batch["CH"] = batch_ch

    return ret_batch
