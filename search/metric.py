import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class DEO():

    def __init__(self):
        pass

    def __call__(self, preds, labels, sensitive_attribute):
        idx_00 = list(set(np.where(sensitive_attribute == 0)
                      [0]) & set(np.where(labels == 0)[0]))
        idx_01 = list(set(np.where(sensitive_attribute == 0)
                      [0]) & set(np.where(labels == 1)[0]))
        idx_10 = list(set(np.where(sensitive_attribute == 1)
                      [0]) & set(np.where(labels == 0)[0]))
        idx_11 = list(set(np.where(sensitive_attribute == 1)
                      [0]) & set(np.where(labels == 1)[0]))

        pred_00 = preds[idx_00]
        pred_01 = preds[idx_01]
        pred_10 = preds[idx_10]
        pred_11 = preds[idx_11]

        gap_0 = pred_00.mean() - pred_10.mean()
        gap_1 = pred_01.mean() - pred_11.mean()
        gap_0 = abs(gap_0)
        gap_1 = abs(gap_1)
 
        gap = gap_0 + gap_1
        return gap


class DSP():
    def __init__(self):
        pass

    def __call__(self, preds, labels, sensitive_attribute):
        idx_0 = np.where(sensitive_attribute == 0)[0]
        idx_1 = np.where(sensitive_attribute == 1)[0]

        labels[idx_0]
        labels[idx_1]

        pred_0 = preds[idx_0]
        pred_1 = preds[idx_1]

        gap = pred_0.mean() - pred_1.mean()
        gap = abs(gap)
        return gap


def pgd_whitebox(model,
                    X,
                    y,
                    use_autocast,
                    epsilon=0.031,
                    num_steps=20,
                    step_size=0.003,
                    random=True,
                    device='cuda'):

    scaler = torch.cuda.amp.GradScaler()

    if use_autocast:
        with torch.cuda.amp.autocast():
            out = model(X)
    else:
        out = model(X)

    err = (out.data.max(1)[1] != y.data).float().sum().cpu().numpy()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(
            *X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        if use_autocast:
            with torch.cuda.amp.autocast():
                with torch.enable_grad():
                    loss = nn.CrossEntropyLoss()(model(X_pgd), y)

            scaler.scale(loss).backward()

        else:
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)

            loss.backward()

        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    if use_autocast:
        with torch.cuda.amp.autocast():
            pred_adv = model(X_pgd)
    else:
        pred_adv = model(X_pgd)

    err_pgd = (pred_adv.data.max(1)[1] != y.data).float().sum().cpu().numpy()
    return err, err_pgd
