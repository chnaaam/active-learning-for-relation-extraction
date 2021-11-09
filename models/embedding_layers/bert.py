import torch.nn as nn

from transformers import BertModel

MODEL_NAME = "monologg/kobert"

class BertEmbeddingLayer(nn.Module):
    def __init__(self, parameters):
        super(BertEmbeddingLayer, self).__init__()

        self.build_layer(parameters)

    def __str__(self):
        return "bert"

    def build_layer(self, parameters):
        self.bert = BertModel.from_pretrained(MODEL_NAME)

    def forward(self, tokens, parameters):
        features = self.bert(
            tokens,
            token_type_ids=parameters["token_type_ids"],
            attention_mask=parameters["attention_mask"])

        return features[0]

