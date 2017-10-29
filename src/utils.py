from torch.autograd import Variable
from collections import Iterable
import constants


def argsort(x, reverse=True):
    return sorted(range(len(x)),
                  key=lambda y: len(x.__getitem__(y)),
                  reverse=reverse)


def repackage_hidden(h):
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def seqdropout(x, keep_prob=0.5):
    mask = np.random.binomial(n=1, p=(1 - keep_prob), size=x.shape)
    mask[:, 0] = 0
    out = x.copy()
    out[np.where(mask == 1)] = constants.UNK

    return out


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


class Node(object):

    def __init__(self, state, logprob, tokenid, alpha=0.65, parent=None):
        self.state = state
        if parent is None:
            self.depth = 1
            self.logprob = logprob / self.depth**alpha
            self.true_logprob = logprob
        else:
            self.depth = parent.depth + 1
            self.logprob = parent.logprob * parent.depth**alpha + logprob
            self.logprob = self.logprob / self.depth**alpha
            self.true_logprob = parent.true_logprob + logprob

        self.tokenid = tokenid
        self.parent = parent
