import torch.nn as nn

from transformers import ElectraModel

MODEL_NAME = "monologg/koelectra-base-v3-discriminator"

class ElectraEmbeddingLayer(nn.Module):
    def __init__(self, parameters):
        super(ElectraEmbeddingLayer, self).__init__()

        self.build_layer(parameters)

    def __str__(self):
        return "electra"

    def build_layer(self, parameters):
        self.electra = ElectraModel.from_pretrained(MODEL_NAME)

    def forward(self, tokens, parameters):
        features = self.electra(
            tokens,
            token_type_ids=parameters["token_type_ids"],
            attention_mask=parameters["attention_mask"])

        return features[0]

