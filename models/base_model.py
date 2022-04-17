import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def __call__(self, input):
        raise NotImplementedError("Please Implement this method") 