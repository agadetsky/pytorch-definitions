import torch
import torch.nn as nn
import torch.nn.functional as F
import constants


class Input(nn.Module):
    """
    Class for Input or Input - Adaptive or dummy conditioning
    """

    def __init__(self):
        super(Input, self).__init__()

    def forward(self, x):
        """
        Vectors are already prepaired in DataLoaders, so just return them
        """
        return x


class InputAttention(nn.Module):
    """
    Class for Input Attention conditioning
    """

    def __init__(self, n_attn_tokens, n_attn_embsize,
                 n_attn_hid, attn_dropout, sparse=False):
        super(InputAttention, self).__init__()
        self.n_attn_tokens = n_attn_tokens
        self.n_attn_embsize = n_attn_embsize
        self.n_attn_hid = n_attn_hid
        self.attn_dropout = attn_dropout
        self.sparse = sparse

        self.embs = nn.Embedding(
            num_embeddings=self.n_attn_tokens,
            embedding_dim=self.n_attn_embsize,
            padding_idx=constants.PAD_IDX,
            sparse=self.sparse
        )

        self.ann = nn.Sequential(
            nn.Dropout(p=self.attn_dropout),
            nn.Linear(
                in_features=self.n_attn_embsize,
                out_features=self.n_attn_hid
            ),
            nn.Tanh()
        )  # maybe use ReLU or other?

        self.a_linear = nn.Linear(
            in_features=self.n_attn_hid,
            out_features=self.n_attn_embsize
        )

    def forward(self, word, context):
        x_embs = self.embs(word)
        mask = self.get_mask(context)
        return mask * x_embs

    def get_mask(self, context):
        context_embs = self.embs(context)
        lengths = (context != constants.PAD_IDX)
        for_sum_mask = lengths.unsqueeze(2).float()
        lengths = lengths.sum(1).float().view(-1, 1)
        logits = self.a_linear(
            (self.ann(context_embs) * for_sum_mask).sum(1) / lengths
        )
        return F.sigmoid(logits)


class CharCNN(nn.Module):
    """
    Class for CH conditioning
    """

    def __init__(self, n_ch_tokens, ch_maxlen, ch_emb_size,
                 ch_feature_maps, ch_kernel_sizes):
        super(CharCNN, self).__init__()
        assert len(ch_feature_maps) == len(ch_kernel_sizes)

        self.n_ch_tokens = n_ch_tokens
        self.ch_maxlen = ch_maxlen
        self.ch_emb_size = ch_emb_size
        self.ch_feature_maps = ch_feature_maps
        self.ch_kernel_sizes = ch_kernel_sizes

        self.feature_mappers = nn.ModuleList()
        for i in range(len(self.ch_feature_maps)):
            reduced_length = self.ch_maxlen - self.ch_kernel_sizes[i] + 1
            self.feature_mappers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=self.ch_feature_maps[i],
                        kernel_size=(
                            self.ch_kernel_sizes[i],
                            self.ch_emb_size
                        )
                    ),
                    nn.Tanh(),
                    nn.MaxPool2d(kernel_size=(reduced_length, 1))
                )
            )

        self.emb = nn.Embedding(
            self.n_ch_tokens,
            self.ch_emb_size,
            padding_idx=constants.PAD_IDX
        )

    def forward(self, x):
        # x - [batch_size x maxlen]
        bsize, length = x.size()
        assert length == self.ch_maxlen
        x_embs = self.emb(x).view(bsize, 1, self.ch_maxlen, self.ch_emb_size)

        cnn_features = []
        for i in range(len(self.ch_feature_maps)):
            cnn_features.append(
                self.feature_mappers[i](x_embs).view(bsize, -1)
            )

        return torch.cat(cnn_features, dim=1)


class Hidden(nn.Module):
    """
    Class for Hidden conditioning
    """

    def __init__(self, cond_size, hidden_size, out_size):
        super(Hidden, self).__init__()
        self.cond_size = cond_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.in_size = self.cond_size + self.hidden_size
        self.linear = nn.Linear(
            in_features=self.in_size,
            out_features=self.out_size
        )

    def forward(self, hidden, conds):
        seqlen = hidden.size(1)  # batch_first=True
        repeated_conds = conds.view(-1).repeat(seqlen)
        repeated_conds = repeated_conds.view(seqlen, *conds.size())
        repeated_conds = repeated_conds.permute(
            1, 0, 2
        )  # batchsize x seqlen x cond_dim
        concat = torch.cat(
            [repeated_conds, hidden], dim=2
        )  # concat by last dim
        return F.tanh(self.linear(concat))


class Gated(nn.Module):
    """
    Class for Gated conditioning
    """

    def __init__(self, cond_size, hidden_size):
        super(Gated, self).__init__()
        self.cond_size = cond_size
        self.hidden_size = hidden_size
        self.in_size = self.cond_size + self.hidden_size
        self.linear1 = nn.Linear(
            in_features=self.in_size,
            out_features=self.hidden_size
        )
        self.linear2 = nn.Linear(
            in_features=self.in_size,
            out_features=self.cond_size
        )
        self.linear3 = nn.Linear(
            in_features=self.in_size,
            out_features=self.hidden_size
        )

    def forward(self, hidden, conds):
        seqlen = hidden.size(1)  # batch_first=True
        repeated_conds = conds.view(-1).repeat(seqlen)
        repeated_conds = repeated_conds.view(seqlen, *conds.size())
        repeated_conds = repeated_conds.permute(
            1, 0, 2
        )  # batchsize x seqlen x cond_dim
        concat = torch.cat(
            [repeated_conds, hidden], dim=2
        )  # concat by last dim
        z_t = F.sigmoid(self.linear1(concat))
        r_t = F.sigmoid(self.linear2(concat))
        masked_concat = torch.cat(
            [repeated_conds * r_t, hidden], dim=2
        )
        hat_s_t = F.tanh(self.linear3(masked_concat))
        return (1 - z_t) * hidden + z_t * hat_s_t
