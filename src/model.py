import constants
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from EmbeddingSoftAttention import EmbeddingSoftAttention


class BaseModel(nn.Module):

    def __init__(
        self, ntokens=100000, nx=300, nhid=600,
        ncond=300, nlayers=3, dropout=0.5,
    ):
        super(BaseModel, self).__init__()

        self.keep_prob = dropout
        self.ntokens = ntokens
        self.nx = nx
        self.ncond = ncond
        self.nhid = nhid
        self.n_rnn_input = nx + ncond
        self.nlayers = nlayers

        self.dropout = nn.Dropout(p=self.keep_prob)

        self.embs = nn.Embedding(num_embeddings=self.ntokens,
                                 embedding_dim=self.nx,
                                 padding_idx=constants.PAD)

        self.rnn = nn.LSTM(input_size=self.n_rnn_input,
                           hidden_size=self.nhid,
                           num_layers=self.nlayers,
                           batch_first=True,
                           dropout=self.keep_prob)

        self.linear = nn.Linear(in_features=self.nhid,
                                out_features=self.ntokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embs.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size=96, cuda=True):
        hiddens = [
            Variable(torch.zeros(batch_size, self.nlayers, self.nhid)),
            Variable(torch.zeros(batch_size, self.nlayers, self.nhid)),
        ]
        if cuda:
            for i in range(2):
                hiddens[i] = hiddens[i].cuda()
        return hiddens

    def forward(self, x, lengths, maxlen, conds, hidden):

        for i in range(len(hidden)):
            hidden[i] = hidden[i].permute(1, 0, 2).contiguous()

        assert x.size(0) == conds.size(0)
        assert self.ncond == conds.size(1)
        assert x.size(0) == lengths.size(0)

        lengths = lengths.cpu().data.numpy()

        embs = self.embs(x)
        repeated_conds = conds.view(-1).repeat(maxlen)
        repeated_conds = repeated_conds.view(maxlen, *conds.size())
        repeated_conds = repeated_conds.permute(1, 0, 2)

        embs = torch.cat([repeated_conds, embs], dim=-1)
        embs = pack(embs, lengths, batch_first=True)

        output, hidden = self.rnn(embs, hidden)
        output = unpack(output, batch_first=True)[0]
        output = self.dropout(output)

        padded_output = Variable(
            torch.zeros(output.size()[0], maxlen, output.size()[2])
        ).cuda()

        padded_output[:, :max(lengths), :] = output

        decoded = self.linear(
            padded_output.view(
                padded_output.size(0) * padded_output.size(1),
                padded_output.size(2),
            )
        )

        decoded = decoded.view(
            padded_output.size(0),
            padded_output.size(1),
            decoded.size(1),
        )

        hidden = list(hidden)
        for i in range(len(hidden)):
            hidden[i] = hidden[i].permute(1, 0, 2).contiguous()

        return decoded, hidden


class Model(nn.Module):

    def __init__(
        self, ntokens=100000, nx=300, nhid=600,
        ncond=300, nlayers=3, dropout=0.5,
        use_attention=True, n_att_tokens=1000000,
        n_att_hid=256, learn_attention_embeddings=False,
    ):
        super(Model, self).__init__()

        self.keep_prob = dropout
        self.ntokens = ntokens
        self.nx = nx
        self.ncond = ncond
        self.nhid = nhid
        self.n_rnn_input = nx + ncond
        self.nlayers = nlayers
        self.use_attention = use_attention
        self.learn_attention_embeddings = learn_attention_embeddings
        self.n_att_tokens = n_att_tokens
        self.n_att_hid = n_att_hid

        self.dropout = nn.Dropout(p=self.keep_prob)

        self.embs = nn.Embedding(num_embeddings=self.ntokens,
                                 embedding_dim=self.nx,
                                 padding_idx=constants.PAD)

        self.rnn = nn.LSTM(input_size=self.n_rnn_input,
                           hidden_size=self.nhid,
                           num_layers=self.nlayers,
                           batch_first=True,
                           dropout=self.keep_prob)

        self.linear = nn.Linear(in_features=self.nhid,
                                out_features=self.ntokens)

        if self.use_attention:
            self.att = EmbeddingSoftAttention(
                ntokens=self.n_att_tokens,
                ncond=self.ncond,
                nhid=self.n_att_hid,
                learn_embs=self.learn_attention_embeddings
            )

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embs.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size=96, cuda=True):
        hiddens = [
            Variable(torch.zeros(batch_size, self.nlayers, self.nhid)),
            Variable(torch.zeros(batch_size, self.nlayers, self.nhid)),
        ]
        if cuda:
            for i in range(2):
                hiddens[i] = hiddens[i].cuda()
        return hiddens

    def forward(self, x, lengths, maxlen, conds, contexts, hidden):

        for i in range(len(hidden)):
            hidden[i] = hidden[i].permute(1, 0, 2).contiguous()

        assert x.size(0) == conds.size(0)
        assert self.ncond == conds.size(1)
        assert x.size(0) == lengths.size(0)

        lengths = lengths.cpu().data.numpy()

        embs = self.embs(x)
        if self.use_attention:
            conds = self.att(conds, contexts)
        repeated_conds = conds.view(-1).repeat(maxlen)
        repeated_conds = repeated_conds.view(maxlen, *conds.size())
        repeated_conds = repeated_conds.permute(1, 0, 2)

        embs = torch.cat([repeated_conds, embs], dim=-1)
        embs = pack(embs, lengths, batch_first=True)

        output, hidden = self.rnn(embs, hidden)
        output = unpack(output, batch_first=True)[0]
        output = self.dropout(output)

        padded_output = Variable(
            torch.zeros(output.size()[0], maxlen, output.size()[2])
        ).cuda()

        padded_output[:, :max(lengths), :] = output

        decoded = self.linear(
            padded_output.view(
                padded_output.size(0) * padded_output.size(1),
                padded_output.size(2),
            )
        )

        decoded = decoded.view(
            padded_output.size(0),
            padded_output.size(1),
            decoded.size(1),
        )

        hidden = list(hidden)
        for i in range(len(hidden)):
            hidden[i] = hidden[i].permute(1, 0, 2).contiguous()

        return decoded, hidden
