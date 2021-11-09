import torch.nn as nn

from transformers import DistilBertModel

MODEL_NAME = "monologg/distilkobert"

class DistilBertEmbeddingLayer(nn.Module):
    def __init__(self, parameters):
        super(DistilBertEmbeddingLayer, self).__init__()

        self.build_layer(parameters)

    def __str__(self):
        return "bert"

    def build_layer(self, parameters):
        self.distil_bert = DistilBertModel.from_pretrained(MODEL_NAME)

    def forward(self, tokens, parameters):
        features = self.distil_bert(
            tokens,
            # token_type_ids=parameters["token_type_ids"],
            attention_mask=parameters["attention_mask"])

        return features[0]

