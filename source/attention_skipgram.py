from .layers import InputAttention
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionSkipGram(nn.Module):

    def __init__(self, n_attn_tokens, n_attn_embsize,
                 n_attn_hid, attn_dropout, sparse=False):
        super(AttentionSkipGram, self).__init__()
        self.n_attn_tokens = n_attn_tokens
        self.n_attn_embsize = n_attn_embsize
        self.n_attn_hid = n_attn_hid
        self.attn_dropout = attn_dropout
        self.sparse = sparse

        self.emb0_lookup = InputAttention(
            n_attn_tokens=self.n_attn_tokens,
            n_attn_embsize=self.n_attn_embsize,
            n_attn_hid=self.n_attn_hid,
            attn_dropout=self.attn_dropout,
            sparse=self.sparse
        )
        self.emb1_lookup = nn.Embedding(
            num_embeddings=self.n_attn_tokens,
            embedding_dim=self.n_attn_embsize,
            sparse=self.sparse
        )
        self.emb1_lookup.weight.data.zero_()

    def forward(self, words, context, neg):
        idx = torch.LongTensor(words.size(0), 1).random_(
            0, context.size(1)
        ).to(words.device)
        labels = context.gather(1, idx).squeeze(1)

        w_embs = self.emb0_lookup(words, context)
        c_embs = self.emb1_lookup(labels)
        n_embs = self.emb1_lookup(neg)

        pos_ips = torch.sum(w_embs * c_embs, 1)
        neg_ips = torch.bmm(
            n_embs, torch.unsqueeze(w_embs, 1).permute(0, 2, 1)
        ).squeeze(2)

        # Neg Log Likelihood
        pos_loss = -torch.mean(F.logsigmoid(pos_ips))
        neg_loss = -torch.mean(F.logsigmoid(-neg_ips).sum(1))

        return pos_loss + neg_loss
