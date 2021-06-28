import torch.nn as nn
import torch.nn.functional as F
import torch 

class AuxLoss(nn.Module):
    """CityScapes only"""

    def __init__(self, aux_weight=0.4):
        super(AuxLoss, self).__init__()
        self.aux_weight = aux_weight  # aux_lossの重み

    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs['out'], targets, ignore_index=255, reduction='mean')
        loss_aux = F.cross_entropy(outputs['aux'], targets, ignore_index=255, reduction='mean')

        return loss+self.aux_weight*loss_aux

class simpleLoss(nn.Module):
    """CityScapes only"""
    def __init__(self):
        super(simpleLoss, self).__init__()
    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs['out'], targets, ignore_index=255, reduction='mean')
        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs['out'], targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()