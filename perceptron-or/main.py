import numpy as np

# np.random.seed(0)

train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_data_y = np.array([-1, 1, 1, 1])


def threshold_activation_function(x):
    return -1 if x < 0 else 1


# Add the bias to the input data.
# size = 4x3
train_data = np.insert(train_data, 0, 1, axis=1)

epochs = 1000
random_weight_threshold = 0.5
learning_rate = 0.001

# Synapses that will be learned.
# size = 3x1
weigths = np.random.uniform(-random_weight_threshold, random_weight_threshold, 3)

for epoch in range(epochs):
    # Go through all the data.
    for t in zip(train_data, train_data_y):
        # Predict the y-value based on the weights.
        predicted_y = threshold_activation_function(weigths.dot(t[0]))
        # The size of the input data is the size of the weights that must be updated.
        for entry in range(3):
            weigths[entry] = weigths[entry] - (
                learning_rate * (predicted_y - t[1]) * t[0][entry]
            )

# Test on training data if the output is correct.
# In this case, there is no data to test.
for t in zip(train_data, train_data_y):
    predicted_y = threshold_activation_function(weigths.dot(t[0]))
    print(t)
    print(predicted_y)
