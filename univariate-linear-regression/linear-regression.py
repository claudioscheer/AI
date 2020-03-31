#!../bin/python3
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("dataset.csv", delimiter=",")
alpha = 0.01
t0 = 0
t1 = 0
iterations = 100000
m = len(data)

# normalize data between 0 and 1
x = [(j - np.amin(data[::, 0])) / (np.amax(data[::, 0]) - np.amin(data[::, 0]))
     for j in data[::, 0]]
y = [(j - np.amin(data[::, 2])) / (np.amax(data[::, 2]) - np.amin(data[::, 2]))
     for j in data[::, 2]]


def hypothesis(x):
    return t0 + t1 * x


def squared_error():
    return (1 / (2 * m)) * sum([(hypothesis(x[i]) - y[i]) ** 2 for i in range(m)])


# plt.ion()
# plt.scatter(x, y, color='b', marker='x')
# line1, = plt.plot(x, [hypothesis(x[i]) for i in range(m)], 'r-')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# train data
i = 0
while i < iterations:

    for d in range(m):
        gradient = (hypothesis(x[d]) - y[d]) * x[d]
        # gradient_t1 = (hypothesis(x[d]) - y[d]) * x[d]

        temp_t0 = t0 - alpha * gradient
        temp_t1 = t1 - alpha * gradient

        t0 = temp_t0
        t1 = temp_t1
        # line1.set_ydata([hypothesis(x[i]) for i in range(m)])
        # plt.draw()
        i += 1
        print(f"Error: {squared_error()} | I: {i}")
        # plt.pause(0.00001)

print(f"\nSlope: {t0}\nIntercept(y): {t1}")

plt.scatter(x, y, color='b', marker='x')
plt.plot(x, [hypothesis(x[i]) for i in range(m)], 'r-')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
