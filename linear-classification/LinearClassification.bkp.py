import numpy as np
import pickle
import matplotlib.pyplot as plt

np.random.seed(1)


def unpickle(file):
    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data


def predict(weights, x, bias):
    return np.dot(weigths, x) + bias


def reshape_plot_cifar_image(image, label):
    reshaped_image = np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0))
    plt.imshow(reshaped_image)
    plt.title(label)
    plt.show()


def hinge_loss(images, weights, bias):
    sum_loss = 0
    for x, y in zip(images[b"data"], images[b"labels"]):
        scores = predict(weigths, x, bias)
        margins = np.maximum(0, scores - scores[y] + 1)
        margins[y] = 0
        sum_loss += np.sum(margins)
    return sum_loss / images_per_batch


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def log_likelihood(x_train, y_train, weigths, b):
    loss_sum = 0
    for x, y in zip(x_train, y_train):
        x_normalized = x / np.linalg.norm(x)
        scores = predict(weigths, x_normalized, bias)
        scores_softmax = softmax(scores)
        loss_sum += -np.log(scores_softmax[y])
    return loss_sum / images_per_batch


batches_meta = unpickle("data/batches.meta")
images_per_batch = batches_meta[b"num_cases_per_batch"]
label_names = batches_meta[b"label_names"]
data_1 = unpickle("data/data_batch_1")

# test_index = 5
# test_image = data_1[b"data"][test_index]
# test_label = label_names[data_1[b"labels"][test_index]]

# reshape_plot_cifar_image(test_image, test_label.decode("UTF-8"))

# The weights that the model will learn to best fit the linear classification.
weigths = np.random.rand(10, 3072)
# The probability that an image belongs to a class. Helps to generalize the model.
bias = np.random.rand(10)
# total_loss = hinge_loss(data_1, weigths, bias)
# print(total_loss)

loss = log_likelihood(data_1[b"data"], data_1[b"labels"], weigths, bias)
print(loss)
