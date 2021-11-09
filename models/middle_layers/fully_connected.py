import torch.nn as nn

class FullyConnectedLayer(nn.Module):
    def __init__(self, parameters):
        super(FullyConnectedLayer, self).__init__()

    def __str__(self):
        return "lstm"

    def build(self, parameters):
        self.fc = nn.Linear(in_features=parameters["fc_in_size"], out_features=parameters["label_size"])

    def forward(self, features, parameters):
        raise NotImplementedError()