import torch.nn as nn

class MiddleLayer(nn.Module):
    def __init__(self, layer, parameters):
        super(MiddleLayer, self).__init__()

        self.middle_layer = layer(parameters)

    def __str__(self):
        return f"Middle Layer Name : {self.middle_layer}"

    def forward(self, features, parameters):
        return self.middle_layer(features, parameters)
