import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, hidden_layers_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers_size[0], hidden_layers_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers_size[1], hidden_layers_size[2]),
            nn.ReLU(),
            nn.Linear(hidden_layers_size[2], 1),
        )

    def forward(self, x):
        out = self.layers(x)
        return out
