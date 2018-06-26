from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm import tqdm
import torch
import numpy as np
from . import constants
from source.datasets import pad


def train_epoch(dataloader, model, optimizer, device, clip_to, logfile):
    """
    Function for training the model one epoch
        dataloader - either LanguageModeling or DefinitionModeling dataloader
        model - DefinitionModelingModel
        optimizer - optimizer to use (usually Adam)
        device - cuda/cpu
        clip_to - value to clip gradients
        logfile - where to log training
    """
    # switch model to training mode
    model.train()
    # train
    mean_batch_loss = 0
    for batch in tqdm(dataloader, file=logfile):
        y_true = torch.from_numpy(batch.pop("y")).to(device).view(-1)
        # prepare model args
        to_input = {"x": torch.from_numpy(batch["x"]).to(device)}
        if not model.params["pretrain"]:
            if model.is_w2v:
                to_input["input"] = torch.from_numpy(batch["input"]).to(device)
            if model.is_ada:
                to_input["input"] = torch.from_numpy(
                    batch["input_adaptive"]
                ).to(device)
            if model.is_attn:
                to_input["word"] = torch.from_numpy(batch["word"]).to(device)
                to_input["context"] = torch.from_numpy(
                    batch["context"]
                ).to(device)
            if model.params["use_ch"]:
                to_input["CH_word"] = torch.from_numpy(
                    batch["CH"]
                ).to(device)

        y_pred, hidden = model(**to_input)
        batch_loss = F.cross_entropy(
            y_pred, y_true,
            ignore_index=constants.PAD_IDX
        )
        optimizer.zero_grad()
        batch_loss.backward()
        clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), clip_to
        )
        optimizer.step()
        logfile.flush()
        mean_batch_loss += batch_loss.item()

    mean_batch_loss = mean_batch_loss / len(dataloader)
    logfile.write(
        "Mean training loss on epoch: {0}\n".format(mean_batch_loss)
    )
    logfile.flush()


def test(dataloader, model, device, logfile):
    """
    Function for testing the model on dataloader
        dataloader - either LanguageModeling or DefinitionModeling dataloader
        model - DefinitionModelingModel
        device - cuda/cpu
        logfile - where to log evaluation
    """
    # switch model to evaluation mode
    model.eval()
    # eval
    lengths_sum = 0
    loss_sum = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, file=logfile):
            y_true = torch.from_numpy(batch.pop("y")).to(device).view(-1)
            # prepare model args
            to_input = {"x": torch.from_numpy(batch["x"]).to(device)}
            if not model.params["pretrain"]:
                if model.is_w2v:
                    to_input["input"] = torch.from_numpy(
                        batch["input"]
                    ).to(device)
                if model.is_ada:
                    to_input["input"] = torch.from_numpy(
                        batch["input_adaptive"]
                    ).to(device)
                if model.is_attn:
                    to_input["word"] = torch.from_numpy(
                        batch["word"]
                    ).to(device)
                    to_input["context"] = torch.from_numpy(
                        batch["context"]
                    ).to(device)
                if model.params["use_ch"]:
                    to_input["CH_word"] = torch.from_numpy(
                        batch["CH"]
                    ).to(device)

            y_pred, hidden = model(**to_input)
            loss_sum += F.cross_entropy(
                y_pred,
                y_true,
                ignore_index=constants.PAD_IDX,
                size_average=False
            ).item()
            lengths_sum += (to_input["x"] != constants.PAD_IDX).sum().item()
            logfile.flush()

    perplexity = np.exp(loss_sum / lengths_sum)
    logfile.write(
        "Perplexity: {0}\n".format(perplexity)
    )
    logfile.flush()
    return perplexity


def generate(model, voc, tau, n, length, device, prefix=None,
             input=None, word=None, context=None, context_voc=None,
             CH_word=None, ch_voc=None):
    """
    model - DefinitionModelingModel
    voc - model Vocabulary
    tau - temperature to generate with
    n - number of samples
    length - length of the sample
    device - cuda/cpu
    prefix - prefix to read until generation
    input - vectors for Input/InputAdaptive conditioning
    word - word for InputAttention conditioning
    context - context for InputAttention conditioning
    context_voc - Vocabulary for InputAttention conditioning
    CH_word - word for CH conditioning
    ch_voc - Vocabulary for CH conditioning
    """
    model.eval()
    to_input = {}
    if not model.params["pretrain"]:
        if model.is_w2v or model.is_ada:
            assert input is not None, ("input argument is required because"
                                       "model uses w2v or adagram vectors")
            assert input.dim() == 1, ("input argument must be vector"
                                      "but its dim is {0}".format(input.dim()))
            to_input["input"] = input.repeat(n).view(n, -1).to(device)
        if model.is_attn:
            assert word is not None, ("word argument is required because"
                                      "model uses attention")
            assert context is not None, ("context argument is required because"
                                         "model uses attention")
            assert context_voc is not None, ("context_voc argument is required"
                                             "because model uses attention")
            assert isinstance(word, str), ("word argument must be string")
            assert isinstance(context, str), ("context argument must be "
                                              "string")
            to_input["word"] = torch.LongTensor(
                [context_voc.encode(word)]
            ).repeat(n).view(n).to(device)
            to_input["context"] = torch.LongTensor(
                context_voc.encode_seq(context.split())
            ).repeat(n).view(n, -1).to(device)
        if model.params["use_ch"]:
            assert CH_word is not None, ("CH_word argument is required because"
                                         "because model uses CH conditioning")
            assert ch_voc is not None, ("ch_voc argument is required because"
                                        "because model uses CH conditioning")
            assert isinstance(CH_word, str), ("CH_word must be string")
            to_input["CH_word"] = torch.LongTensor(
                pad(
                    [constants.BOS_IDX] +
                    ch_voc.encode_seq(list(CH_word)) +
                    [constants.EOS_IDX], ch_voc.tok_maxlen + 2,
                    constants.PAD_IDX
                )
            ).repeat(n).view(n, -1).to(device)

    to_input["x"] = None
    to_input["hidden"] = None  # pytorch automatically init to zeroes
    ret = [[] for i in range(n)]
    if prefix is not None:
        assert isinstance(prefix, str), "prefix argument must be string"
        if len(prefix.split()) > 0:
            to_input["x"] = torch.LongTensor(
                voc.encode_seq(prefix.split())
            ).repeat(n).view(n, -1).to(device)
    else:
        to_input["x"] = torch.randint(
            model.params["ntokens"], size=(1, ), dtype=torch.long
        ).repeat(n).view(n, -1).to(device)
        prefix = voc.decode(to_input["x"][0][0].item())
    with torch.no_grad():
        for i in range(length):
            output, to_input["hidden"] = model(**to_input)
            output = output.view((n, -1, model.params["ntokens"]))[:, -1, :]
            to_input["x"] = F.softmax(
                output / tau, dim=1
            ).multinomial(num_samples=1)
            for i in range(n):
                ret[i].append(to_input["x"][i][0].item())

    output = [[] for i in range(n)]
    for i in range(n):
        decoded = voc.decode_seq(ret[i])
        for j in range(length):
            if decoded[j] == constants.EOS:
                break
            output[i].append(decoded[j])
        output[i] = " ".join(map(str, output[i]))

    return "\n".join(output)
