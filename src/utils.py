from torch.autograd import Variable
from collections import Iterable
import constants


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
