import numpy as np


# load data set
data_set = np.loadtxt(open("data_set.csv", "rb"),
                      delimiter=",", skiprows=1).astype("float")
x = np.array(data_set[:, [0, 1]])
y = np.array(data_set[:, [2]])

# normalize data
x = x / np.amax(x, axis=0)
y = y / np.amax(y, axis=0)


class NeuralNetwork:
    # define neural network
    def __init__(self):
        # define hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        self.w1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
        self.w2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, x):
        # propagate inputs through network
        self.z2 = np.dot(x, self.w1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        y_predicted = self.sigmoid(self.z3)
        return y_predicted

    def sigmoid(self, z):
        # apply sigmoid activation function
        return 1 / (1 + np.exp(-z))


nn = NeuralNetwork()
y_predicted = nn.forward(x)
print(y_predicted)
