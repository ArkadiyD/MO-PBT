import torch
import torch.nn as nn

import torch

from TRADES.trades import trades_loss


class SPLoss(nn.Module):
    def __init__(self, label_name='labels', logits_name='logits'):
        super(SPLoss, self).__init__()

        self.label_name = label_name
        self.logits_name = logits_name

    def forward(self, logits, target, sensitive_attribute):
        preds = torch.softmax(logits, dim=1)

        priv = preds[sensitive_attribute.bool(), 1]
        unpriv = preds[~sensitive_attribute.bool(), 1]
        value = torch.abs(torch.mean(priv)-torch.mean(unpriv))
        return value


class CESPLoss(nn.Module):
    def __init__(self, weight=None, mu=0.0):
        super(CESPLoss, self).__init__()

        self.weight = weight
        self.mu = mu

        if weight is not None:
            self.loss1 = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            self.loss1 = torch.nn.CrossEntropyLoss()

        self.loss2 = SPLoss()

    def forward(self, logits, target, sensitive_attribute):
        if self.mu == 0.0:
            return self.loss1(logits, target)

        return self.loss1(logits, target) + self.mu * self.loss2(logits, target, sensitive_attribute)


class ApplyTradesLoss(nn.Module):
    def __init__(self, beta, perturb_steps, use_autocast):
        super(ApplyTradesLoss, self).__init__()

        self.beta = beta
        self.step_size = 0.003
        self.epsilon = 0.031
        self.perturb_steps = perturb_steps
        self.distance = 'l_inf'
        self.use_autocast = use_autocast

    def forward(self, model, x_natural, target, optimizer):
        return trades_loss(model=model,
                           x_natural=x_natural,
                           y=target,
                           optimizer=optimizer,
                           step_size=self.step_size,
                           epsilon=self.epsilon,
                           perturb_steps=self.perturb_steps,
                           beta=self.beta,
                           distance=self.distance,
                           use_autocast=self.use_autocast)
