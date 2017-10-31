import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constants


class EmbeddingSoftAttention(nn.Module):

    def __init__(self, ntokens, ncond, nhid, learn_embs=True):
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
        batch_size = x.size(0)
        x_embs = self.embs(x)
        masks = []
        for i in range(batch_size):
            context_embs_i = self.embs(context[i])
            masks.append(
                F.sigmoid(
                    self.a_linear(
                        self.ann(context_embs_i).mean(0)
                    )
                )
            )
        return torch.cat(masks, 0).view(batch_size, self.ncond) * x_embs
