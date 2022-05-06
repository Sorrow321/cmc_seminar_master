import timm
import torch.nn as nn
from .base_model import BaseModel

class EffNetV2Model(BaseModel):
    def __init__(self, inp_size, *args):
        super().__init__()
        self.model = timm.create_model(*args, pretrained=True, num_classes=0)
        if inp_size != 0:
            self.lin = nn.Linear(inp_size, 1024)
            self.transform = True
        else:
            self.transform = False

    def __call__(self, input):
        x = self.model(input)
        if self.transform:
            x = self.lin(x)
        return x