import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import BostonDataset
from network import NeuralNetwork
import progressbar


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
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=test_sampler)

model = NeuralNetwork([8, 32, 32, 16])
model.cuda()

loss_function = nn.MSELoss()
optimization_function = torch.optim.Adam(
    model.parameters(), lr=0.01, betas=(0.99, 0.995), weight_decay=0.1
)

model.train()
num_epochs = 1000
for _ in progressbar.progressbar(range(num_epochs)):
    for batch_index, (x, y) in enumerate(train_loader):
        optimization_function.zero_grad()
        y_predicted = model(x)
        error = loss_function(y_predicted.float(), y)
        error.backward()
        optimization_function.step()

model.eval()
with torch.no_grad():
    # predicted_values = torch.tensor([]).cuda()
    # correct_values = torch.tensor([]).cuda()
    for batch_index, (x, y) in enumerate(test_loader):
        y_predicted = model(x)
        error = loss_function(y_predicted.float(), y)
        print(f"Error: {error}; Prediction: {y_predicted}; Correct: {y}")
        # predicted_values = torch.cat((predicted_values, y_predicted), -1)
        # correct_values = torch.cat((correct_values, y), -1)

    # print(predicted_values.cpu().numpy()[0])
    # print(correct_values.cpu().numpy()[0])
    # print(accuracy)
