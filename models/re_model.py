import torch.nn as nn

from . import (
    EmbeddingLayer,
    MiddleLayer,
    OutputLayer
)

class ReModel(nn.Module):
    def __init__(self, embedding_layer, middle_layer, output_layer, **parameters):
        super(ReModel, self).__init__()

        self.embedding_layer = EmbeddingLayer(layer=embedding_layer, parameters=parameters)
        self.middle_layer = MiddleLayer(layer=middle_layer, parameters=parameters)
        self.output_layer = OutputLayer(layer=output_layer, parameters=parameters)

        # self.print_layers()

    def print_layers(self):
        print(f">> Model for Relation Extraction")
        print(f">> {self.embedding_layer}")
        print(f">> {self.middle_layer}")
        print(f">> {self.output_layer}")

    def forward(self, tokens, parameters):
        embed_features = self.embedding_layer(tokens=tokens, parameters=parameters)
        middle_features = self.middle_layer(features=embed_features, parameters=parameters)
        output = self.output_layer(features=middle_features, parameters=parameters)

        return output
