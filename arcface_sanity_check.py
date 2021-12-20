import torch
import torch.nn as nn
import torch.nn.functional as F
from arcface import ArcFace

batch_size = 8
n_features = 64
n_classes = 6
input = torch.randn(batch_size, n_features)
labels = torch.randint(0, n_classes, (batch_size,))

# parameters from the original paper
m = 0.5
s = 64
arcface_layer = ArcFace(n_features, n_classes, m, s)

output = arcface_layer(input)
print(output.shape)
print(output)