import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, layer, parameters):
        super(EmbeddingLayer, self).__init__()
        
        self.embedding_layer = layer(parameters)

    def __str__(self):
        return f"Embedding Layer Name : {self.embedding_layer}"

    def forward(self, tokens, parameters):
        return self.embedding_layer(tokens=tokens, parameters=parameters)