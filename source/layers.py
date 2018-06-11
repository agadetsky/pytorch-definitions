import torch
import torch.nn as nn
import constants


class Input(nn.Module):
    """
    Class for Input or Input - Adaptive or dummy conditioning
    """

    def __init__(self):
        super(Input, self).__init__()

    def forward(self):
        pass


class InputAttention(nn.Module):
    """
    Class for Input Attention conditioning
    """

    def __init__(self):
        super(InputAttention, self).__init__()

    def forward(self):
        pass


class CharCNN(nn.Module):
    """
    Class for CH conditioning
    """

    def __init__(self):
        super(CharCNN, self).__init__()

    def forward(self):
        pass


class Hidden(nn.Module):
    """
    Class for Hidden conditioning
    """

    def __init__(self):
        super(Hidden, self).__init__()

    def forward(self):
        pass


class Gated(nn.Module):
    """
    Class for Gated conditioning
    """

    def __init__(self):
        super(Gated, self).__init__()

    def forward(self):
        pass
