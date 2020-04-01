import torch
import torch.nn as nn

train_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).cuda()
train_data_y = torch.tensor([[0], [1], [1], [1]], dtype=torch.float).cuda()


class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.layers = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


model = Perceptron()
model.cuda()

loss_function = torch.nn.L1Loss()
optimization_function = torch.optim.SGD(model.parameters(), lr=0.03)

model.train()

for _ in range(10000):
    optimization_function.zero_grad()
    y_predicted = model(train_data)
    loss = loss_function(y_predicted.float(), train_data_y)
    loss.backward()
    optimization_function.step()

model.eval()
y_predicted = model(train_data)
after_train_error = loss_function(y_predicted.float(), train_data_y)
print(after_train_error)
print(y_predicted)
