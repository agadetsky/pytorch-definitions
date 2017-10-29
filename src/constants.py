# FOR MODEL VOCABULARY

PAD = 0
UNK = 1
BOS = 2
EOS = 3
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

# GPU SUPPORT

use_cuda = True
cuda_devices = [0, 1, 2]

# FOR OPTIMIZER

opt_method = "Adam"
opt_initial_lr = 0.001
opt_decay_factor = 0.1
opt_decay_patience = 0
opt_grad_clip = 5
