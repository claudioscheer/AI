from numpy import *

# based on MILONE author
def correlation_classification(correlation_coeficient):
    if correlation_coeficient <= 0.5:
        return "Inappropriate"
    elif correlation_coeficient <= 0.6:
        return "Terrible"
    elif correlation_coeficient <= 0.7:
        return "Poor"
    elif correlation_coeficient <= 0.8:
        return "Aceptable"
    elif correlation_coeficient <= 0.9:
        return "Good"
    else:
        return "Great"

# n * E[xy] - E[x]E[y] / sqrt(n * E[x^2] - E[x]^2) sqrt(n * E[y^2] - E[y]^2)
def correlation_coeficient_pearson(points):
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
    return v1 / (sqrt(v2) * sqrt(v3))


if __name__ == "__main__":
    points = genfromtxt("data.csv", delimiter=";")
    print correlation_classification(correlation_coeficient_pearson(points))
