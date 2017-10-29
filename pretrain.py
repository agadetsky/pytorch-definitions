import src.constants as constants
from src.Datasets import WikiText, batchify
from src.data_workflow import Word2Vec
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

TRAIN_DATA = "./data/wikitext-103/wiki.train.tokens"
VAL_DATA = "./data/wikitext-103/wiki.valid.tokens"
TEST_DATA = "./data/wikitext-103/wiki.test.tokens"
INIT_WV = True
WV_WEIGHTS = "./data/w2v_embeddings/GoogleNews-vectors-negative300.bin"
SEED = 42
CUDA = True
BATCH_SIZE = 32
SEQLEN = 30
NCOND = 300
NX = 300
NHID = NX + NCOND
NUM_EPOCHS = 1
NLAYERS = 3
DROPOUT_PROB = 0.5
INITIAL_LR = 0.001
DECAY_FACTOR = 0.1
DECAY_PATIENCE = 0
GRAD_CLIP = 5
MODEL_CKPT = "./pretrain_wiki_exp/best_pretrain"
TRAIN = True
SAVE_VOCAB_TO = "./pretrain_wiki_exp/wiki_vocab.json"

LOGFILE = open('./pretrain_wiki_exp/log.txt', 'a')
EXP_RESULTS = open("./pretrain_wiki_exp/results.txt", "a")

# code start

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

tqdm.write("Reading and tokenizing Wiki!", file=LOGFILE)
LOGFILE.flush()

wiki = WikiText(
    TRAIN_DATA,
    VAL_DATA,
    TEST_DATA
)

wiki.vocab.save(SAVE_VOCAB_TO)

if TRAIN:

    if INIT_WV:
        tqdm.write("Reading Word2Vec for init embeddings!", file=LOGFILE)
        LOGFILE.flush()

        w2v = Word2Vec(
            WV_WEIGHTS
        )

    tqdm.write("Initialising BaseModel", file=LOGFILE)
    LOGFILE.flush()

    net = BaseModel(
        ntokens=len(wiki.vocab.i2w),
        nx=NX,
        nhid=NHID,
        ncond=NCOND,
        nlayers=NLAYERS,
        dropout=DROPOUT_PROB,
    )

    if INIT_WV:

        tqdm.write("Initialising Encoder weights!", file=LOGFILE)
        LOGFILE.flush()

        embs_init = net.embs.weight.data.numpy()
        for word in wiki.vocab.w2i.keys():
            if word in w2v.w2i:
                embs_init[wiki.vocab.w2i[word]] = w2v.get_cond_vector(word)

        net.embs.weight.data.copy_(torch.from_numpy(embs_init))

    # if cuda
    net.cuda()
    ##########

    tqdm.write("Initialising Criterion and Optimizer!", file=LOGFILE)
    LOGFILE.flush()

    # if cuda
    criterion = CrossEntropyLoss(ignore_index=constants.PAD).cuda()
    #########
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
                len(wiki.train) / ((SEQLEN + 1) * BATCH_SIZE)
            )
        )
        with tqdm(total=num_batches, file=LOGFILE) as pbar:
            train_iter = batchify(wiki.train, wiki.vocab, SEQLEN, BATCH_SIZE)
            for batch_x, batch_y in train_iter:
                hidden = net.init_hidden(
                    batch_x.shape[0], cuda=True)  # if cuda
                scheduler.optimizer.zero_grad()

                lengths = (batch_x != constants.PAD).sum(axis=1)
                maxlen = int(max(lengths))
                lengths = Variable(torch.from_numpy(lengths)).cuda().long()
                batch_x = Variable(torch.from_numpy(batch_x)).cuda()
                conds = Variable(
                    torch.from_numpy(np.zeros((batch_x.size(0), NCOND)))
                )
                conds = conds.cuda().float()
                batch_y = Variable(torch.from_numpy(batch_y)).cuda().view(-1)

                output, hidden = net(batch_x, lengths, maxlen, conds, hidden)
                loss = criterion(output.view(-1, len(wiki.vocab.i2w)), batch_y)
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
                len(wiki.val) / ((SEQLEN + 1) * BATCH_SIZE)
            )
        )
        with tqdm(total=num_batches, file=LOGFILE) as pbar:
            val_iter = batchify(wiki.val, wiki.vocab, SEQLEN, BATCH_SIZE)
            for batch_x, batch_y in val_iter:
                hidden = net.init_hidden(
                    batch_x.shape[0], cuda=True)  # if cuda

                lengths = (batch_x != constants.PAD).sum(axis=1)
                maxlen = int(max(lengths))
                lengths = Variable(torch.from_numpy(lengths)).cuda().long()
                batch_x = Variable(torch.from_numpy(batch_x)).cuda()
                conds = Variable(
                    torch.from_numpy(np.zeros((batch_x.size(0), NCOND)))
                )
                conds = conds.cuda().float()
                batch_y = Variable(torch.from_numpy(batch_y)).cuda().view(-1)

                output, hidden = net(batch_x, lengths, maxlen, conds, hidden)
                loss = criterion(output.view(-1, len(wiki.vocab.i2w)), batch_y)

                lengths_cnt += lengths.sum().cpu().data.numpy()[0]
                loss_i.append(loss.data.cpu().numpy()[
                              0] * lengths.sum().float())

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
        ntokens=len(wiki.vocab.i2w),
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

test_iter = batchify(wiki.test, wiki.vocab, SEQLEN, BATCH_SIZE)
lengths_cnt = 0
num_batches = int(
    np.ceil(
        len(wiki.test) / ((SEQLEN + 1) * BATCH_SIZE)
    )
)
loss_i = []
with tqdm(total=num_batches, file=LOGFILE) as pbar:
    for batch_x, batch_y in test_iter:
        hidden = net.init_hidden(
            batch_x.shape[0], cuda=True
        )  # if cuda

        lengths = (batch_x != constants.PAD).sum(axis=1)
        maxlen = int(max(lengths))
        lengths = Variable(torch.from_numpy(lengths)).cuda().long()
        batch_x = Variable(torch.from_numpy(batch_x)).cuda()
        conds = Variable(
            torch.from_numpy(np.zeros((batch_x.size(0), NCOND)))
        )
        conds = conds.cuda().float()
        batch_y = Variable(torch.from_numpy(batch_y)).cuda().view(-1)

        output, hidden = net(batch_x, lengths, maxlen, conds, hidden)
        loss = cross_entropy(
            output.view(-1, len(wiki.vocab.i2w)),
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
tqdm.write("INIT_WV = {0}".format(INIT_WV), file=EXP_RESULTS)
tqdm.write("WV_WEIGHTS = {0}".format(WV_WEIGHTS), file=EXP_RESULTS)
tqdm.write("SEED = {0}".format(SEED), file=EXP_RESULTS)
tqdm.write("CUDA = {0}".format(CUDA), file=EXP_RESULTS)
tqdm.write("BATCH_SIZE = {0}".format(BATCH_SIZE), file=EXP_RESULTS)
tqdm.write("SEQLEN = {0}".format(SEQLEN), file=EXP_RESULTS)
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
tqdm.write("TRAIN = {0}".format(TRAIN), file=EXP_RESULTS)
tqdm.write("SAVE_VOCAB_TO = {0}\n\n".format(SAVE_VOCAB_TO), file=EXP_RESULTS)
tqdm.write("RESULTS:\n", file=EXP_RESULTS)
if TRAIN:
    tqdm.write("TRAIN PPL: {0}".format(
        train_losses[min_val_loss_idx]), file=EXP_RESULTS
    )
    tqdm.write("VAL PPL: {0}".format(min_val_loss), file=EXP_RESULTS)
tqdm.write("TEST PPL: {0}".format(test_loss), file=EXP_RESULTS)

EXP_RESULTS.flush()
EXP_RESULTS.close()
