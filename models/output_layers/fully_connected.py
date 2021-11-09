import torch.nn as nn

class FullyConnectedLayer(nn.Module):
    def __init__(self, parameters):
        super(FullyConnectedLayer, self).__init__()

        self.build_layer(parameters)

    def __str__(self):
        return "lstm"

    def build_layer(self, parameters):
        self.fc = nn.Linear(in_features=parameters["fc_in_size"], out_features=parameters["label_size"])

    def forward(self, features, parameters):
        return self.fc(features[:, 0, :])