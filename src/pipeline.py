from torch.autograd import Variable
from utils import Node
import constants
import torch
import numpy as np


def greedy_sample(
        model, dictionary, cond_vec,
        start=constants.BOS_WORD, end=constants.EOS_WORD,
        length=20, tau=1e-6):

    model.eval()

    hidden = model.init_hidden(batch_size=1)
    cond = Variable(torch.from_numpy(cond_vec)).cuda().float().view(1, -1)
    maxlen = 1
    lengths = Variable(torch.LongTensor([1, ])).cuda()

    end_id = dictionary.encode(end)
    cur_id = dictionary.encode(start)
    definition = [cur_id, ]
    logprob = 0
    counter = 0
    while cur_id != end_id:
        x_i = Variable(torch.LongTensor([[cur_id]])).cuda()

        output, hidden = model(
            x=x_i,
            lengths=lengths,
            maxlen=maxlen,
            conds=cond,
            hidden=hidden,
        )

        prob_i = torch.nn.functional.softmax(output[0, 0] / tau)
        true_prob = torch.nn.functional.softmax(output[0, 0])

        cur_id = torch.multinomial(prob_i)
        cur_id = int(cur_id.cpu().data.numpy()[0])
        definition.append(cur_id)
        logprob += np.log(true_prob[cur_id].cpu().data.numpy()[0])

        counter += 1
        if counter > length:
            break

    return (
        " ".join([dictionary.decode(i) for i in definition]),
        logprob,
    )


def beam_sample(
        model, dictionary, cond_vec, beam_width=16,
        start=constants.BOS_WORD, end=constants.EOS_WORD,
        length=20, tau=None, alpha=0.65):

    model.eval()
    hidden_init = model.init_hidden(batch_size=1)
    num_states = len(hidden_init)
    cond_vec = np.tile(cond_vec, beam_width).reshape(beam_width, -1)
    cond = Variable(torch.from_numpy(cond_vec)).cuda().float()
    lengths = Variable(torch.LongTensor([1 for _ in range(beam_width)])).cuda()
    maxlen = 1

    end_id = dictionary.encode(end)
    cur_id = dictionary.encode(start)

    cur_nodes = [
        Node(
            state=hidden_init,
            logprob=0,
            tokenid=cur_id,
            parent=None,
            alpha=alpha,
        )
    ]
    result = []

    for _ in range(length):
        bsize = len(cur_nodes)
        xs = []
        hiddens = [[] for i in range(num_states)]
        for node in cur_nodes:
            xs.append(node.tokenid)
            for i in range(num_states):
                hiddens[i].append(node.state[i])

        xs = np.array(xs, dtype=int).reshape((-1, 1))
        xs = Variable(torch.from_numpy(xs)).cuda()
        for i in range(num_states):
            hiddens[i] = torch.cat(hiddens[i], dim=0)

        output, hidden = model(
            x=xs,
            lengths=lengths[:bsize],
            maxlen=maxlen,
            conds=cond[:bsize],
            hidden=hiddens,
        )

        if tau is None:
            true_probs = torch.nn.functional.softmax(output[:, 0, :])
            true_probs = true_probs.cpu().data.numpy()
            samples = np.argsort(true_probs)[:, ::-1]
            samples = samples[:, :beam_width]
        else:
            probs = torch.nn.functional.softmax(output[:, 0, :] / tau)
            true_probs = torch.nn.functional.softmax(output[:, 0, :])
            true_probs = true_probs.cpu().data.numpy()
            samples = torch.multinomial(probs, beam_width, True)
            samples = samples.cpu().data.numpy()

        next_cur_nodes = []
        for i in range(bsize):
            for j in range(beam_width):

                state = []
                for k in range(num_states):
                    state.append(
                        hidden[k][i].view(1, *hidden[k][i].size())
                    )

                cur_node = Node(
                    state=state,
                    logprob=np.log(true_probs[i, samples[i, j]]),
                    tokenid=samples[i, j],
                    parent=cur_nodes[i],
                    alpha=alpha
                )
                if samples[i, j] == end_id:
                    result.append(cur_node)
                else:
                    next_cur_nodes.append(cur_node)
        cur_nodes = sorted(next_cur_nodes, key=lambda x: -x.logprob)
        cur_nodes = cur_nodes[:beam_width]
        if len(cur_nodes) == 0:
            break

    beams = sorted(result, key=lambda x: -x.logprob)

    decoded = []
    visited = set()

    for beam in beams:
        lst = []
        cur = beam
        while cur is not None:
            lst.append(cur.tokenid)
            cur = cur.parent
        lst = lst[::-1]
        sentence = " ".join([dictionary.decode(word) for word in lst])
        if sentence not in visited:
            visited.add(sentence)
            decoded.append([sentence, beam.logprob, beam.true_logprob])
    return decoded
