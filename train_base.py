import src.constants as constants
from src.Datasets import Definitions, batchify_defs
from src.data_workflow import Word2Vec, AdaGram, Zeros
from src.model import BaseModel
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.nn.utils import clip_grad_norm
import numpy as np
from tqdm import tqdm

# parameters

TRAIN_DATA = "./data/main_data/definitions_train.json"
VAL_DATA = "./data/main_data/definitions_val.json"
TEST_DATA = "./data/main_data/definitions_test.json"
INIT_MODEL_CKPT = "./pretrain_wiki_exp/best_pretrain"  # or None
MODEL_VOCAB = "./pretrain_wiki_exp/wiki_vocab.json"  # or None
COND_TYPE = 0  # 0 for Zeros, 1 for Word2Vec, 2 for AdaGram
COND_WEIGHTS = None # None for Zeros or Path for Word2Vec and AdaGram
FIX_EMBEDDINGS = False
SEED = 42
CUDA = True
BATCH_SIZE = 16
NCOND = 300
NX = 300
NHID = NX + NCOND
NUM_EPOCHS = 35
NLAYERS = 3
DROPOUT_PROB = 0.5
INITIAL_LR = 0.001
DECAY_FACTOR = 0.1
DECAY_PATIENCE = 0
GRAD_CLIP = 5
MODEL_CKPT = "./train_def_zeros_exp/best_train_type_{0}".format(COND_TYPE)
TRAIN = True

LOGFILE = open('./train_def_zeros_exp/log.txt', 'a')
EXP_RESULTS = open("./train_def_zeros_exp/results.txt", "a")

# code start

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


tqdm.write("Reading Data!", file=LOGFILE)
LOGFILE.flush()

defs = Definitions(
    train=TRAIN_DATA,
    val=VAL_DATA,
    test=TEST_DATA,
    with_examples=False,
    vocab_path=MODEL_VOCAB
)

tqdm.write("Reading vectors for conditioning!", file=LOGFILE)
LOGFILE.flush()

if COND_TYPE == 0:
    cond = Zeros(NCOND)
elif COND_TYPE == 1:
    cond = Word2Vec(COND_WEIGHTS)
elif COND_TYPE == 2:
    import julia
    j = julia.Julia()
    cond = AdaGram(COND_WEIGHTS)
else:
    raise ValueError("No such COND_TYPE = {0}".format(COND_TYPE))

if TRAIN:
    tqdm.write("Initialising Model!", file=LOGFILE)
    LOGFILE.flush()

    net = BaseModel(
        ntokens=len(defs.vocab.i2w),
        nx=NX,
        nhid=NHID,
        ncond=NCOND,
        nlayers=NLAYERS,
        dropout=DROPOUT_PROB,
    )

    net.cuda()  # if cuda
    if INIT_MODEL_CKPT is not None:
        params = torch.load(INIT_MODEL_CKPT)
        net.load_state_dict(params["state_dict"])

    net.embs.weight.requires_grad = FIX_EMBEDDINGS

    tqdm.write("Initialising Criterion and Optimizer!", file=LOGFILE)
    LOGFILE.flush()

    criterion = CrossEntropyLoss(ignore_index=constants.PAD).cuda()

    scheduler = ReduceLROnPlateau(
        torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=INITIAL_LR
        ),
        factor=DECAY_FACTOR,
        patience=DECAY_PATIENCE
    )

    tqdm.write("Start training!", file=LOGFILE)
    LOGFILE.flush()

    min_val_loss = np.inf
    min_val_loss_idx = None
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(NUM_EPOCHS), file=LOGFILE):
        LOGFILE.flush()
        ### train ###
        net.train()
        lengths_cnt = 0
        loss_i = []
        num_batches = int(
            np.ceil(
                len(defs.train) / BATCH_SIZE
            )
        )
        with tqdm(total=num_batches, file=LOGFILE) as pbar:
            train_iter = batchify_defs(
                defs.train, defs.vocab, cond, BATCH_SIZE
            )
            for batch_x, batch_y, conds in train_iter:
                hidden = net.init_hidden(
                    batch_x.shape[0], cuda=True
                )  # if cuda
                scheduler.optimizer.zero_grad()

                lengths = (batch_x != constants.PAD).sum(axis=1)
                maxlen = int(max(lengths))
                lengths = Variable(torch.from_numpy(lengths)).cuda().long()
                batch_x = Variable(torch.from_numpy(batch_x)).cuda()
                conds = Variable(
                    torch.from_numpy(conds)
                )
                conds = conds.cuda().float()
                batch_y = Variable(torch.from_numpy(batch_y)).cuda().view(-1)

                output, hidden = net(batch_x, lengths, maxlen, conds, hidden)
                loss = criterion(output.view(-1, len(defs.vocab.i2w)), batch_y)
                loss.backward()
                clip_grad_norm(net.parameters(), GRAD_CLIP)
                scheduler.optimizer.step()

                lengths_cnt += lengths.sum().cpu().data.numpy()[0]
                loss_i.append(
                    loss.data.cpu().numpy()[0] * lengths.sum().float()
                )

                pbar.update(1)
                LOGFILE.flush()

            train_losses.append(
                np.exp(
                    np.sum(loss_i).cpu().data.numpy()[0] / lengths_cnt
                )
            )

        tqdm.write(
            "Epoch: {0}, Train PPL: {1}".format(epoch, train_losses[-1]),
            file=LOGFILE
        )
        LOGFILE.flush()

        ### val ###
        net.eval()
        loss_i = []
        lengths_cnt = 0
        num_batches = int(
            np.ceil(
                len(defs.val) / BATCH_SIZE
            )
        )
        with tqdm(total=num_batches, file=LOGFILE) as pbar:
            val_iter = batchify_defs(
                defs.val, defs.vocab, cond, BATCH_SIZE
            )
            for batch_x, batch_y, conds in val_iter:
                hidden = net.init_hidden(
                    batch_x.shape[0], cuda=True
                )  # if cuda

                lengths = (batch_x != constants.PAD).sum(axis=1)
                maxlen = int(max(lengths))
                lengths = Variable(torch.from_numpy(lengths)).cuda().long()
                batch_x = Variable(torch.from_numpy(batch_x)).cuda()
                conds = Variable(
                    torch.from_numpy(conds)
                )
                conds = conds.cuda().float()
                batch_y = Variable(torch.from_numpy(batch_y)).cuda().view(-1)

                output, hidden = net(batch_x, lengths, maxlen, conds, hidden)
                loss = criterion(output.view(-1, len(defs.vocab.i2w)), batch_y)

                lengths_cnt += lengths.sum().cpu().data.numpy()[0]
                loss_i.append(
                    loss.data.cpu().numpy()[0] * lengths.sum().float()
                )

                pbar.update(1)
                LOGFILE.flush()

            val_losses.append(
                np.exp(np.sum(loss_i).cpu().data.numpy()[0] / lengths_cnt)
            )

        scheduler.step(metrics=val_losses[-1])

        tqdm.write(
            "Epoch: {0}, Val PPL: {1}".format(epoch, val_losses[-1]),
            file=LOGFILE
        )
        LOGFILE.flush()

        if val_losses[-1] < min_val_loss:
            min_val_loss = val_losses[-1]
            min_val_loss_idx = epoch
            torch.save({"state_dict": net.state_dict()}, MODEL_CKPT)

if not TRAIN:

    tqdm.write("Loading Model weights for testing!", file=LOGFILE)
    LOGFILE.flush()

    net = BaseModel(
        ntokens=len(defs.vocab.i2w),
        nx=NX,
        nhid=NHID,
        ncond=NCOND,
        nlayers=NLAYERS,
        dropout=DROPOUT_PROB,
    )

    net.cuda()  # if cuda
    params = torch.load(MODEL_CKPT)
    net.load_state_dict(params["state_dict"])

tqdm.write("Testing...", file=LOGFILE)
LOGFILE.flush()

net.eval()

test_iter = batchify_defs(defs.test, defs.vocab, cond, BATCH_SIZE)
lengths_cnt = 0
num_batches = int(
    np.ceil(
        len(defs.test) / BATCH_SIZE
    )
)
loss_i = []
with tqdm(total=num_batches, file=LOGFILE) as pbar:
    for batch_x, batch_y, conds in test_iter:
        hidden = net.init_hidden(
            batch_x.shape[0], cuda=True
        )  # if cuda

        lengths = (batch_x != constants.PAD).sum(axis=1)
        maxlen = int(max(lengths))
        lengths = Variable(torch.from_numpy(lengths)).cuda().long()
        batch_x = Variable(torch.from_numpy(batch_x)).cuda()
        conds = Variable(
            torch.from_numpy(conds)
        )
        conds = conds.cuda().float()
        batch_y = Variable(torch.from_numpy(batch_y)).cuda().view(-1)

        output, hidden = net(batch_x, lengths, maxlen, conds, hidden)
        loss = cross_entropy(
            output.view(-1, len(defs.vocab.i2w)),
            batch_y,
            ignore_index=constants.PAD
        )

        lengths_cnt += lengths.sum().cpu().data.numpy()[0]
        loss_i.append(loss.data.cpu().numpy()[0] * lengths.sum().float())

        pbar.update(1)
        LOGFILE.flush()

test_loss = np.exp(np.sum(loss_i).cpu().data.numpy()[0] / lengths_cnt)
tqdm.write("Test PPL: {0}".format(test_loss), file=LOGFILE)
LOGFILE.flush()
LOGFILE.close()


tqdm.write("Parameters:\n", file=EXP_RESULTS)
tqdm.write("TRAIN_DATA = {0}".format(TRAIN_DATA), file=EXP_RESULTS)
tqdm.write("VAL_DATA = {0}".format(VAL_DATA), file=EXP_RESULTS)
tqdm.write("TEST_DATA = {0}".format(TEST_DATA), file=EXP_RESULTS)
tqdm.write("INIT_MODEL_CKPT = {0}".format(INIT_MODEL_CKPT), file=EXP_RESULTS)
tqdm.write("MODEL_VOCAB = {0}".format(MODEL_VOCAB), file=EXP_RESULTS)
tqdm.write("COND_TYPE = {0}".format(COND_TYPE), file=EXP_RESULTS)
tqdm.write("COND_WEIGHTS = {0}".format(COND_WEIGHTS), file=EXP_RESULTS)
tqdm.write("FIX_EMBEDDINGS = {0}".format(FIX_EMBEDDINGS), file=EXP_RESULTS)
tqdm.write("SEED = {0}".format(SEED), file=EXP_RESULTS)
tqdm.write("CUDA = {0}".format(CUDA), file=EXP_RESULTS)
tqdm.write("BATCH_SIZE = {0}".format(BATCH_SIZE), file=EXP_RESULTS)
tqdm.write("NCOND = {0}".format(NCOND), file=EXP_RESULTS)
tqdm.write("NX = {0}".format(NX), file=EXP_RESULTS)
tqdm.write("NHID = {0}".format(NCOND), file=EXP_RESULTS)
tqdm.write("NUM_EPOCHS = {0}".format(NUM_EPOCHS), file=EXP_RESULTS)
tqdm.write("NLAYERS = {0}".format(NLAYERS), file=EXP_RESULTS)
tqdm.write("DROPOUT_PROB = {0}".format(DROPOUT_PROB), file=EXP_RESULTS)
tqdm.write("INITIAL_LR = {0}".format(INITIAL_LR), file=EXP_RESULTS)
tqdm.write("DECAY_FACTOR = {0}".format(DECAY_FACTOR), file=EXP_RESULTS)
tqdm.write("DECAY PATIENCE = {0}".format(DECAY_PATIENCE), file=EXP_RESULTS)
tqdm.write("GRAD_CLIP = {0}".format(GRAD_CLIP), file=EXP_RESULTS)
tqdm.write("MODEL_CKPT = {0}".format(MODEL_CKPT), file=EXP_RESULTS)
tqdm.write("TRAIN = {0}\n\n".format(TRAIN), file=EXP_RESULTS)
tqdm.write("RESULTS:\n", file=EXP_RESULTS)
if TRAIN:
    tqdm.write("TRAIN PPL: {0}".format(
        train_losses[min_val_loss_idx]), file=EXP_RESULTS
    )
    tqdm.write("VAL PPL: {0}".format(min_val_loss), file=EXP_RESULTS)
tqdm.write("TEST PPL: {0}".format(test_loss), file=EXP_RESULTS)

EXP_RESULTS.flush()
EXP_RESULTS.close()
