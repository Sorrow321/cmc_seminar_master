import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, in_features, n_classes, m, s):
        super(ArcFace, self).__init__()
        self.m = m
        self.s = s
        self.n_classes = n_classes
        k = torch.sqrt(torch.tensor(1 / in_features))
        self.weight = nn.Parameter(
            2 * k * torch.rand(n_classes, in_features) - k,
            requires_grad=True
        )

    def forward(self, input, labels=None):
        # normalize inputs, W and perform FC
        x = F.linear(
            F.normalize(input),
            F.normalize(self.weight)
        )
        if labels is None:
            # inference mode
            return x
        print(labels)
        # for numerical stability
        x = x.clamp(-1, 1)
        # apply arccos
        x = torch.acos(x)
        # get indexes for labels
        idxes = F.one_hot(labels, self.n_classes) == 1
        print(idxes)
        print(x)
        # add margin
        x[idxes] += self.m
        print(x)
        # return back to cos
        x = torch.cos(x)
        # rescale
        x = x * self.s
        return x