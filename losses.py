# ------------------------------------------------------------------------
# UniMoCo
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Definition of the Unified Contrastive Loss
"""
from torch import nn
import torch
class UnifiedContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(UnifiedContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1)
        sum_pos = (y_true * torch.exp(-y_pred)).sum(1)
        loss = torch.log(1 + sum_neg * sum_pos)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss
