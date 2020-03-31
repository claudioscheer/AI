import numpy as np
import pickle
import matplotlib.pyplot as plt

# np.random.seed(1)


def unpickle(file):
    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data


def predict(weights, x, bias):
    return np.dot(weights, x) + bias


def show_image(image, label):
    reshaped_image = np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0))
    plt.imshow(reshaped_image)
    plt.title(label)
    plt.show()


def hinge_loss(images, weights, bias):
    """SVM loss function."""
    sum_loss = 0
    for x, y in zip(images[b"data"], images[b"labels"]):
        scores = predict(weights, x, bias)
        margins = np.maximum(0, scores - scores[y] + 1)
        margins[y] = 0
        sum_loss += np.sum(margins)
    return sum_loss / images_per_batch


def gradient_descent(weights):
    h = 0.0001
    gradients = ((weigths + h) - weigths) / h
    return gradients


batches_meta = unpickle("data/batches.meta")
images_per_batch = batches_meta[b"num_cases_per_batch"]
label_names = batches_meta[b"label_names"]
data_1 = unpickle("data/data_batch_1")
learning_rate = 0.001

# test_index = 5
# test_image = data_1[b"data"][test_index]
# test_label = label_names[data_1[b"labels"][test_index]]
# show_image(test_image, test_label.decode("UTF-8"))

# for x in range(10):
# The weights that the model will learn to best fit the linear classification.
weigths = np.random.rand(10, 3072)
# The probability that an image belongs to a class. Helps to generalize the model.
bias = np.random.rand(10)
total_loss = hinge_loss(data_1, weigths, bias)
gradients = gradient_descent(weigths)
print(weigths)
print(gradients)
