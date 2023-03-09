import numpy as np
import random
import pandas as pd

# This function makes 10 possible slopes using 10 different samples.
# The return will be a list with the mean result of the slope for every sample.


def derivative():
    # Import the data from the csv
    c = pd.read_csv("data.csv")
    y = c["y_train"]
    x = c["x_test"]
    # Takes random samples and gives the mean slope for every sample.
    result = []
    for i in range(10):
        dx = np.array(random.sample(x.tolist(), 10))
        dy = np.array(random.sample(y.tolist(), 10))
        c = []
        for j in range(len(dx)):
            r = dy[j] / dx[j]
            c.append(r)
        mean = sum(c) / len(c)
        result.append(mean)
    return result

# This function will return you the best result to your dataset.


def slope():
    # Import the results for the data created.
    result_list = derivative()
    c = pd.read_csv("data.csv")
    x = c["x_test"]
    # Empty lists.
    y_list = []
    error = []
    final = []
    result = 0
    # This iteration takes the value of X and estimates the value of Y
    # Multiply X with the list of derivatives and the return its the estimation of Y
    for i in range(len(result_list)):
        r = x * result_list[i]
        y_list.append(r)
    # This iteration gives the total sum of error square for every derivative.
    for y in range(len(y_list)):
        r = (y_list[y] - y) ** 2
        r = sum(r)
        error.append(r)
    # This sort the results in order to return the best result.
    # If the result its more than 3.5, its divided by two.
    for s in range(len(error)):
        if result < error[s]:
            result = error[s]
            final = result_list[s]
    if final >= 3.5:
        final = final * 0.50
    return final


def slope_result():
    # Import the results for the data created.
    d = slope()
    c = pd.read_csv("data.csv")
    x = c["x_test"]
    b = []
    for i in range(len(x)):
        r = d * x[i]
        b.append(r)
    return b
