from transformers import AutoFeatureExtractor, SwinPreTrainedModel, SwinModel
from .base_model import BaseModel


class SwinEmbedding(SwinPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.swin = SwinModel(config)
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.swin(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        return pooled_output



class SwinTransformerModel(BaseModel):
    def __init__(self, *args):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(*args)
        self.model = SwinEmbedding.from_pretrained(*args)
    
    def __call__(self, input):
        inputs = self.feature_extractor(images=input, return_tensors="pt")
        return self.model(**inputs)