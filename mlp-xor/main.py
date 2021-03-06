import torch
import torch.nn as nn

train_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).cuda()
train_data_y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float).cuda()


class MultilayerPerceptron(nn.Module):
    def __init__(self, hidden_layer_size):
        super(MultilayerPerceptron, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, hidden_layer_size), nn.ReLU(), nn.Linear(hidden_layer_size, 1)
        )

    def forward(self, x):
        return self.layers(x)


model = MultilayerPerceptron(10)
model.cuda()

loss_function = nn.MSELoss()
optimization_function = torch.optim.Adam(
    model.parameters(), lr=0.01, betas=(0.99, 0.995), weight_decay=0
)

model.train()

loss_values = []
for _ in range(5000):
    optimization_function.zero_grad()
    y_predicted = model(train_data)
    loss = loss_function(y_predicted.float(), train_data_y)
    loss_values.append(loss.item())
    
    loss.backward()
    optimization_function.step()

model.eval()
y_predicted = model(train_data)
after_train_error = loss_function(y_predicted.float(), train_data_y)
print(after_train_error)
print(y_predicted)
