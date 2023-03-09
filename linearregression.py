import numpy as np
import pandas as pd


def linear_regression():
    c = pd.read_csv("data.csv")
    y = c["y_train"]
    x = c["x_test"]
    data = []
    coefficient = np.polyfit(x, y, 1)
    for i in range(len(x)):
        w = coefficient[0]
        m = coefficient[1]
        result = (w * x[i]) + m
        data.append(result)
    return data


def linear_regression2():
    c = pd.read_csv("data.csv")
    y = c["y_train"]
    x = c["x_test"]
    data = []
    coefficient = np.polyfit(x, y, 2)
    for i in range(len(x)):
        w2 = coefficient[0] ** 2
        w = coefficient[1]
        m = coefficient[2]
        result = (w2 * x[i]) + (w*x[i]) + m
        data.append(result)
    return data


def linear_regression3():
    c = pd.read_csv("data.csv")
    y = c["y_train"]
    x = c["x_test"]
    data = []
    coefficient = np.polyfit(x, y, 3)
    for i in range(len(x)):
        w3 = coefficient[0] ** 3
        w2 = coefficient[1] ** 2
        w = coefficient[2]
        m = coefficient[3]
        result = (w3 * x[i]) + (w2 * x[i]) + (w*x[i]) + m
        data.append(result)
    return data
