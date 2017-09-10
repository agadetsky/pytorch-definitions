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
        self.learn_embs = learn_embs

        if self.learn_embs:
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
        if self.learn_embs:
            x_embs = self.embs(x)
        else:
            x_embs = x
        masks = []
        for i in range(batch_size):
            if self.learn_embs:
                context_embs_i = self.embs(context[i])
            else:
                context_embs_i = context[i]
            masks.append(
                F.sigmoid(
                    self.a_linear(
                        self.ann(context_embs_i).mean(0)
                    )
                )
            )
        return torch.cat(masks, 0).view(batch_size, self.ncond) * x_embs


esa = EmbeddingSoftAttention(100, 300, 256, False)
x = Variable(torch.randn(2, 300))
context = [
    Variable(torch.randn(3, 300)),
    Variable(torch.randn(5, 300))
]

output = esa(x, context)
print(output.size())
