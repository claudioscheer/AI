import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from pad import PadCollate
from dataset import BostonDataset
from network import NeuralNetwork

dataset = BostonDataset("dataset.csv")
dataset_size = len(dataset)
dataset_indices = list(range(dataset_size))

batch_size = 16
test_split = int(np.floor(0.2 * dataset_size))  # 20%
# shuffle data
np.random.shuffle(dataset_indices)

train_indices, test_indices = (
    dataset_indices[test_split:],
    dataset_indices[:test_split],
)

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler
)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=test_sampler
)

model = NeuralNetwork([8, 16, 8])
model.cuda()

loss_function = nn.MSELoss()
optimization_function = torch.optim.Adam(
    model.parameters(), lr=0.01, betas=(0.99, 0.995), weight_decay=0
)

model.train()
num_epochs = 10
for _ in range(num_epochs):
    for batch_index, (x, y) in enumerate(train_loader):
        optimization_function.zero_grad()
        y_predicted = model(x)
        loss = loss_function(y_predicted.float(), y)
        loss.backward()
        optimization_function.step()
