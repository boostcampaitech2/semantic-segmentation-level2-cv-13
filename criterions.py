import torch
import torch.nn as nn
import torch.nn.functional as F

class custom_CrossEntropyLoss(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.CEL = nn.CrossEntropyLoss()


    def forward(self, pred, target):

        if isinstance(pred, list):
            loss = 0
            weights = [0.4, 1]
            assert len(weights) == len(pred)
            for i in range(len(pred)):
                loss += self.CEL(pred[i], target) * weights[i]
            return loss

        else:
            return self.CEL(pred, target)