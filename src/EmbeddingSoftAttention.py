import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constants


class EmbeddingSoftAttention(nn.Module):

    def __init__(self, ntokens, ncond, nhid):
        super(EmbeddingSoftAttention, self).__init__()
        self.ntokens = ntokens
        self.ncond = ncond
        self.nhid = nhid

        self.embs = nn.Embedding(
            num_embeddings=ntokens,
            embedding_dim=ncond,
            padding_idx=constants.PAD
        )

        self.ann = nn.Sequential(
            nn.Linear(
                in_features=self.ncond,
                out_features=self.nhid
            ),
            nn.ReLU()
        )

        self.a_linear = nn.Linear(
            in_features=self.nhid,
            out_features=self.ncond
        )

    def forward(self, x, context):
        x_embs = self.embs(x)
        context_embs = self.embs(context)
        lengths = (context != constants.PAD)
        for_sum_mask = lengths.unsqueeze(2).float()
        lengths = lengths.sum(1).float().view(-1, 1)
        mask = F.sigmoid(
            self.a_linear(
                (self.ann(context_embs) * for_sum_mask).sum(1) / lengths
            )
        )
        return mask * x_embs
