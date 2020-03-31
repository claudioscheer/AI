#!../bin/python3
import numpy as np


def correlation_coeficient_pearson(points):
    """http://www.statisticshowto.com/wp-content/uploads/2012/10/pearson.gif"""
    sum_x = 0
    sum_y = 0
    sum_xy = 0
    sum_x2 = 0
    sum_y2 = 0
    length = len(points)
    for i in range(0, length):
        x = points[i, 0]
        y = points[i, 1]
        sum_x += x
        sum_y += y
        sum_xy += (x*y)
        sum_x2 += x*x
        sum_y2 += y*y
    v1 = (length * sum_xy) - (sum_x * sum_y)
    v2 = (length * sum_x2) - pow(sum_x, 2)
    v3 = (length * sum_y2) - pow(sum_y, 2)
    return v1 / (np.sqrt(v2) * np.sqrt(v3))


if __name__ == "__main__":
    points = np.loadtxt("dataset.csv", delimiter=",")
    print(correlation_coeficient_pearson(points[::, [0, 2]]))
