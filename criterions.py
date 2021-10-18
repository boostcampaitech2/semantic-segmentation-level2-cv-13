import torch
import torch.nn as nn
import torch.nn.functional as F

class custom_CrossEntropyLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.CEL = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.CEL(pred, target)