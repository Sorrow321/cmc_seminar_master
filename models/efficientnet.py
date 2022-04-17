from efficientnet_pytorch import EfficientNet
from .base_model import BaseModel


class EffNetModel(BaseModel):
    def __init__(self, *args):
        super().__init__()
        self.model = EfficientNet.from_pretrained(*args)
    
    def __call__(self, input):
        x = self.model.extract_features(input)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x