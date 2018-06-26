import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from . import constants
from layers import Input, InputAttention, CharCNN, Hidden, Gated


class DefinitionModelingModel(nn.Module):
    """Definition modeling class"""

    def __init__(self, params):
        super(DefinitionModelingModel, self).__init__()
        self.params = params

        self.embs = nn.Embedding(
            num_embeddings=self.params["ntokens"],
            embedding_dim=self.params["nx"],
            padding_idx=constants.PAD_IDX
        )
        self.dropout = nn.Dropout(p=self.params["rnn_dropout"])

        self.n_rnn_input = self.params["nx"]
        if not self.params["pretrain"]:
            self.input_used = self.params["use_input"]
            self.input_used += self.params["use_input_adaptive"]
            self.input_used += self.params["use_input_attention"]
            self.hidden_used = self.params["use_hidden"]
            self.hidden_used += self.params["use_hidden_adaptive"]
            self.hidden_used += self.params["use_hidden_attention"]
            self.gated_used = self.params["use_gated"]
            self.gated_used += self.params["use_gated_adaptive"]
            self.gated_used += self.params["use_gated_attention"]
            self.is_w2v = self.params["use_input"]
            self.is_w2v += self.params["use_hidden"]
            self.is_w2v += self.params["use_gated"]
            self.is_ada = self.params["use_input_adaptive"]
            self.is_ada += self.params["use_hidden_adaptive"]
            self.is_ada += self.params["use_gated_adaptive"]
            self.is_attn = self.params["use_input_attention"]
            self.is_attn += self.params["use_hidden_attention"]
            self.is_attn += self.params["use_gated_attention"]
            self.is_conditioned = self.input_used
            self.is_conditioned += self.hidden_used
            self.is_conditioned += self.gated_used

            # check if either Input* or Hidden/Gated conditioning are used not
            # both
            assert self.input_used + self.hidden_used + \
                self.gated_used <= 1, "Too many conditionings used"

            if not self.is_conditioned and self.params["use_ch"]:
                raise ValueError("Don't use CH conditioning without others")

            self.cond_size = 0
            if self.is_w2v:
                self.input = Input()
                self.cond_size += self.params["input_dim"]
            elif self.is_ada:
                self.input_adaptive = Input()
                self.cond_size += self.params["input_adaptive_dim"]
            elif self.is_attn:
                self.input_attention = InputAttention(
                    n_attn_tokens=self.params["n_attn_tokens"],
                    n_attn_embsize=self.params["n_attn_embsize"],
                    n_attn_hid=self.params["n_attn_hid"],
                    attn_dropout=self.params["attn_dropout"],
                    sparse=self.params["attn_sparse"]
                )
                self.cond_size += self.params["n_attn_embsize"]

            if self.params["use_ch"]:
                self.ch = CharCNN(
                    n_ch_tokens=self.params["n_ch_tokens"],
                    ch_maxlen=self.params["ch_maxlen"],
                    ch_emb_size=self.params["ch_emb_size"],
                    ch_feature_maps=self.params["ch_feature_maps"],
                    ch_kernel_sizes=self.params["ch_kernel_sizes"]
                )
                self.cond_size += sum(self.params["ch_feature_maps"])

            if self.input_used:
                self.n_rnn_input += self.cond_size

            if self.hidden_used:
                self.hidden = Hidden(
                    cond_size=self.cond_size,
                    hidden_size=self.params["nhid"],
                    out_size=self.params["nhid"]
                )
            elif self.gated_used:
                self.gated = Gated(
                    cond_size=self.cond_size,
                    hidden_size=self.params["nhid"]
                )

        self.rnn = nn.LSTM(
            input_size=self.n_rnn_input,
            hidden_size=self.params["nhid"],
            num_layers=self.params["nlayers"],
            batch_first=True,
            dropout=self.params["rnn_dropout"]
        )
        self.linear = nn.Linear(
            in_features=self.params["nhid"],
            out_features=self.params["ntokens"]
        )

        self.init_weights()

    def forward(self, x, input=None, word=None, context=None, CH_word=None, hidden=None):
        """
        x - definitions/LM_sequence to read
        input - vectors for Input, Input-Adaptive or dummy conditioning
        word - words for Input-Attention conditioning
        context - contexts for Input-Attention conditioning
        CH_word - words for CH conditioning
        hidden - hidden states of RNN
        """
        lengths = (x != constants.PAD_IDX).sum(dim=1).detach()
        maxlen = lengths.max().item()
        embs = self.embs(x)
        if not self.params["pretrain"]:
            all_conds = []
            if self.is_w2v:
                all_conds.append(self.input(input))
            elif self.is_ada:
                all_conds.append(self.input_adaptive(input))
            elif self.is_attn:
                all_conds.append(self.input_attention(word, context))
            if self.params["use_ch"]:
                all_conds.append(self.ch(CH_word))
            if self.is_conditioned:
                all_conds = torch.cat(all_conds, dim=1)

            if self.input_used:
                repeated_conds = all_conds.view(-1).repeat(maxlen)
                repeated_conds = repeated_conds.view(maxlen, *all_conds.size())
                repeated_conds = repeated_conds.permute(1, 0, 2)
                embs = torch.cat([repeated_conds, embs], dim=-1)

        embs = pack(embs, lengths, batch_first=True)
        output, hidden = self.rnn(embs, hidden)
        output = unpack(output, batch_first=True)[0]
        output = self.dropout(output)

        if not self.params["pretrain"]:
            if self.hidden_used:
                output = self.hidden(output, all_conds)
            elif self.gated_used:
                output = self.gated(output, all_conds)

        decoded = self.linear(
            output.contiguous().view(
                output.size(0) * output.size(1),
                output.size(2)
            )
        )

        return decoded, hidden

    def init_embeddings(self, freeze):
        initrange = 0.5 / self.params["nx"]
        with torch.no_grad():
            nn.init.uniform_(self.embs.weight, -initrange, initrange)
        self.embs.weight.requires_grad = not freeze

    def init_embeddings_from_pretrained(self, weights, freeze):
        self.embs = self.embs.from_pretrained(weights, freeze)

    def init_rnn(self):
        with torch.no_grad():
            for name, p in self.rnn.named_parameters():
                if "bias" in name:
                    nn.init.constant_(p, 0)
                elif "weight" in name:
                    nn.init.xavier_uniform_(p)

    def init_rnn_from_pretrained(self, weights):
        # k[4:] because we need to remove prefix "rnn." because
        # self.rnn.state_dict() is without "rnn." prefix
        correct_state_dict = {
            k[4:]: v for k, v in weights.items() if k[:4] == "rnn."
        }
        # also we need to correctly initialize weight_ih_l0
        # with pretrained weights because it has different size with
        # self.rnn.state_dict(), other weights has correct shapes if
        # hidden sizes have same shape as in the LM pretraining
        if self.input_used:
            w = torch.empty(4 * self.params["nhid"], self.n_rnn_input)
            nn.init.xavier_uniform_(w)
            w[:, self.cond_size:] = correct_state_dict["weight_ih_l0"]
            correct_state_dict["weight_ih_l0"] = w
        self.rnn.load_state_dict(correct_state_dict)

    def init_linear(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.constant_(self.linear.bias, 0)

    def init_linear_from_pretrained(self, weights):
        # k[7: ] because we need to remove prefix "linear." because
        # self.linear.state_dict() is without "linear." prefix
        self.linear.load_state_dict(
            {k[7:]: v for k, v in weights.items() if k[:7] == "linear."}
        )

    def init_weights(self):
        if self.params["pretrain"]:
            if self.params["w2v_weights"] is not None:
                self.init_embeddings_from_pretrained(
                    torch.load(self.params["w2v_weights"]),
                    self.params["fix_embeddings"]
                )
            else:
                self.init_embeddings(self.params["fix_embeddings"])
            self.init_rnn()
            self.init_linear()
        else:
            if self.params["lm_ckpt"] is not None:
                lm_ckpt_weights = torch.load(self.params["lm_ckpt"])
                lm_ckpt_weights = lm_ckpt_weights["state_dict"]
                self.init_embeddings_from_pretrained(
                    lm_ckpt_weights["embs.weight"],
                    self.params["fix_embeddings"]
                )
                self.init_rnn_from_pretrained(lm_ckpt_weights)
                self.init_linear_from_pretrained(lm_ckpt_weights)
            else:
                if self.params["w2v_weights"] is not None:
                    self.init_embeddings_from_pretrained(
                        torch.load(self.params["w2v_weights"]),
                        self.params["fix_embeddings"]
                    )
                else:
                    self.init_embeddings(self.params["fix_embeddings"])
                self.init_rnn()
                self.init_linear()
            if self.is_attn:
                if self.params["attn_ckpt"] is not None:
                    self.input_attention.init_attn_from_pretrained(
                        torch.load(self.params["attn_ckpt"])["state_dict"],
                        self.params["fix_attn_embeddings"]
                    )
                else:
                    self.input_attention.init_attn(
                        self.params["fix_attn_embeddings"]
                    )
            if self.hidden_used:
                self.hidden.init_hidden()
            if self.gated_used:
                self.gated.init_gated()
            if self.params["use_ch"]:
                self.ch.init_ch()
