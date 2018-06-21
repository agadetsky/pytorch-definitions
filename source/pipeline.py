from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm import tqdm
from constants import PAD_IDX
import torch
import numpy as np


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
        if model.params["pretrain"]:
            if model.cond_size > 0:
                to_input["input"] = torch.from_numpy(
                    batch["dummy_cond"]
                ).to(device)
        else:
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
        batch_loss = F.cross_entropy(y_pred, y_true, ignore_index=PAD_IDX)
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
        logfile - where to log training
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
            if model.params["pretrain"]:
                if model.cond_size > 0:
                    to_input["input"] = torch.from_numpy(
                        batch["dummy_cond"]
                    ).to(device)
            else:
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
                ignore_index=PAD_IDX,
                size_average=False
            ).item()
            lengths_sum += (to_input["x"] != PAD_IDX).sum().item()
            logfile.flush()

    perplexity = np.exp(loss_sum / lengths_sum)
    logfile.write(
        "Perplexity: {0}\n".format(perplexity)
    )
    logfile.flush()
    return perplexity


def generate():
    pass
