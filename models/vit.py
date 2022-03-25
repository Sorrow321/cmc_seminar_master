from transformers import ViTFeatureExtractor, ViTPreTrainedModel, ViTModel
from base_model import BaseModel


class ViTEmbedding(ViTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vit = ViTModel(config, add_pooling_layer=False)
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        return sequence_output[:, 0, :]



class VisionTransformerModel(BaseModel):
    def __init__(self, *args):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(*args)
        self.model = ViTEmbedding.from_pretrained(*args)
    
    def get_embeddings(self, input):
        inputs = self.feature_extractor(images=input, return_tensors="pt")
        return self.model(**inputs)