from efficientnet_pytorch import EfficientNet
from base_model import BaseModel


class EffNetModel(BaseModel):
    def __init__(self, *args):
        self.model = EfficientNet.from_pretrained(*args)
    
    def get_embeddings(self, input):
        x = self.model.extract_features(input)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x