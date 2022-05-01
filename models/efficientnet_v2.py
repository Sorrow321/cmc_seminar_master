import timm
from .base_model import BaseModel

class EffNetV2Model(BaseModel):
    def __init__(self, *args):
        super().__init__()
        self.model = timm.create_model(*args, pretrained=True, num_classes=0)
    
    def __call__(self, input):
        x = self.model(input)
        return x