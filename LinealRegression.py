import numpy as np


def linealregression(x, y):
    data = []
    coeficiente = np.polyfit(x, y, 1)
    for i in range(len(x)):
        w = coeficiente[0]
        m = coeficiente[1]
        result = (w * x[i]) + m
        data.append(result)
    return data


def linealregression2(x, y):
    data = []
    coeficiente = np.polyfit(x, y, 2)
    for i in range(len(x)):
        w2 = coeficiente[0] ** 2
        w = coeficiente[1]
        m = coeficiente[2]
        result = (w2 * x[i]) + (w*x[i]) + m
        data.append(result)
    return data


def linealregression3(x, y):
    data = []
    coeficiente = np.polyfit(x, y, 3)
    for i in range(len(x)):
        w3 = coeficiente[0] ** 3
        w2 = coeficiente[1] ** 2
        w = coeficiente[2]
        m = coeficiente[3]
        result = (w3 * x[i]) + (w2 * x[i]) + (w*x[i]) + m
        data.append(result)
    return data


def error(y, data):
    error = []
    for i in range(len(y)):
        margin = (data[i] * 100) / y[i]
        s = 100 - margin
        error.append(s)
    return error


def meanError(y, data):
    e = []
    for i in range(len(y)):
        margin = (data[i] * 100) / y[i]
        s = 100 - margin
        e.append(s)
    r = sum(e) / len(e)
    return r


def errorsquare(y, data):
    e = []
    for i in range(len(y)):
        margin = (data[i] * 100) / y[i]
        s = (100 - margin) ** 2
        e.append(s)
    return e
