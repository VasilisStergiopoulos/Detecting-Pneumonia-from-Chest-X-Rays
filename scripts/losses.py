import torch
import torch.nn as nn
import torch.nn.functional as F

# Reduced Cross Entropy loss function
class ReducedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(ReducedCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, target, alpha=0):
        ce = self.cross_entropy(pred, target)
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, num_classes=pred.size(1))
        mse =  torch.mean(torch.sum(1 / 2 * (pred - target) ** 2, dim=1))
        loss = ce + alpha * mse
        return loss
    

# Mixed Cross Entropy loss function
class MixedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MixedCrossEntropyLoss, self).__init__()

    # Calculate the loss function value
    def calculate_loss(self, pred, target, alpha=0.5):
        loss = 1 / ((1 - alpha) * (2 - alpha)) * torch.mean(torch.sum(pred ** (2 - alpha) - target ** (2 - alpha), dim=1)) - 1 / (1 - alpha) * torch.mean(torch.sum(target ** (1 - alpha) * (pred - target), dim=1))
        return loss  
    
    # Loss function forward pass
    def forward(self, pred, target, alpha=0.5):
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, num_classes=pred.size(1))
        loss = self.calculate_loss(pred, target, alpha)
        return loss
    
# Define the linear loss function
class LinearLoss(nn.Module):
    def __init__(self):
        super(LinearLoss, self).__init__()

    def forward(self, pred, target, alpha=1):
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, num_classes=pred.size(1))
        loss = alpha / 6 * ((1 - pred) ** 3 - (1 - target) ** 3) + alpha / 2 * (pred - target) * (1 - target) ** 2
        return torch.mean(torch.sum(loss, dim=1))
    

# Piecewise linear loss
class PiecewiseLinearLoss(nn.Module):
    def __init__(self):
        super(PiecewiseLinearLoss, self).__init__()
    
    def forward(self, pred, target, alpha):
        values = torch.tensor([0.5], device=torch.device("cuda:0"))
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, num_classes=pred.size(1))
        with torch.no_grad():
            heavy_pred = torch.heaviside(1 / 2 - pred, values=values)
            heavy_target = torch.heaviside(1 / 2 - target, values=values)
        term_1 = (alpha / 12 - alpha / 2 * target ** 2 + alpha / 3 * target ** 3) * heavy_target
        term_2 = -(alpha / 12 - alpha / 2 * pred ** 2 + alpha / 3 * pred ** 3) * heavy_pred
        term_3 = alpha / 4 * (pred - 1 / 2) * (1 - heavy_pred) + alpha / 4 * (1 / 2 - target) * (1 - heavy_target)
        term_4 = -(pred - target) * (alpha * target * (1 - target) * heavy_target + alpha / 4 * (1 - heavy_target))
        return torch.mean(torch.sum(term_1 + term_2 + term_3 + term_4, dim=1))
