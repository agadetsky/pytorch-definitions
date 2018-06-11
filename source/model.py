import torch
import torch.nn as nn
import constants


class Params():

    def __init__(self):
        # Basic RNN parameters
        self.nx = None
        self.ntokens = None
        self.nlayers = None
        self.nhid = None
        self.rnn_dropout = None
        # Input parameters
        self.use_input = None
        self.input_dim = None
        # Input adaptive parameters
        self.use_input_adaptive = None
        self.input_adaptive_dim = None
        # Input attention parameters
        self.use_input_attention = None
        self.n_attn_tokens = None
        self.n_attn_hid = None
        self.attn_dropout = None
        # CH parameters
        self.use_ch = None
        self.n_ch_tokens = None
        self.ch_maxlen = None
        self.ch_emb_size = None
        self.ch_feature_maps = None
        self.ch_kernel_sizes = None
        # Hidden parameters
        # will be done later
        # Gated parameters
        # will be done later


class DefinitionModelingModel(nn.Module):
    """Definition modeling class"""

    def __init__(self, params):
        super(DefinitionModelingModel, self).__init__()

        self.params = params

    def forward(self):
        pass
