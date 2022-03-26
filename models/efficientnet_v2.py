import timm
from base_model import BaseModel

class EffNetV2Model(BaseModel):
    def __init__(self, *args):
        self.model = timm.create_model(*args, pretrained=True, num_classes=0)
    
    def get_embeddings(self, input):
        x = self.model(input)
        return x