from torch.utils.data import Dataset, DataLoader
import constants
import json


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

    def add_token(self, tok):
        if tok not in self.tok2id:
            self.tok2id[tok] = len(self.tok2id)
            self.id2tok[len(self.id2tok)] = tok

    def save(self, path):
        with open(path, "w") as outfile:
            json.dump(self.id2tok, outfile, indent=4)

    def load(self, path):
        with open(path, "r") as infile:
            self.id2tok = json.load(infile)
        self.id2tok = {int(k): v for k, v in self.id2tok.items()}
        self.tok2id = {}
        for i in self.id2tok.keys():
            self.tok2id[self.id2tok[i]] = i


def LanguageModelingCollate(batch):
    pass


def DefinitionModelingCollate(batch):
    pass


class LanguageModelingDataset(Dataset):
    """LanguageModeling dataset."""

    def __init__(self, file, bptt):
        """
        Args:
            file (string): Path to the file.
            bptt (int): length of one sentence
            vocab (Vocabulary): word vocab to use
        """
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class DefinitionModelingDataset(Dataset):
    """DefinitionModeling dataset."""

    def __init__(self, file, vocab, input_vectors=None,
                 input_adaptive_vectors=None, context_vocab=None,
                 ch_vocab=None):
        """
        Args:
            file (string): Path to the file.
            vocab (Vocabulary): word vocab to use
            input_vectors (ndarray): vectors for Input conditioning
            input_adaptive_vectors (ndarray): vectors for Input-Adaptive conditioning
            context_vocab (Vocabulary): vocab for context words for Input-Attention
            ch_vocab (Vocabulary): char vocab for CH conditioning
        """
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
