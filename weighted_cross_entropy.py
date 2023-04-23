import torch
import torch.nn as nn

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        """
        Args:
            input (torch.Tensor): raw predictions of shape (N, C, H, W), where N is the batch size, C is the number of classes, H is the height, and W is the width.
            target (torch.Tensor): true labels of shape (N, H, W), where N is the batch size, H is the height, and W is the width.
        """
        log_prob = nn.functional.log_softmax(input, dim=1)
        loss = nn.functional.nll_loss(log_prob, target, weight=self.weight, reduction='mean')
        return loss