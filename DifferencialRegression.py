import numpy as np
import LinealRegression as Lr


def differencial(x, y):
    # Stablish the difference between the X
    diff = np.diff(x)
    c = []
    # Make the value of y into the same of the variable diff
    for i in range(1, len(y)):
        c.append(y[i])
    # Calculate the lineal regression of the values.
    c = Lr.linealregression(diff, c)
    # Return the lineal regression approximation of the value of y
    d = [0] * len(x)
    for i in range(1, len(diff)):
        d[i] = c[i] + x[i - 1]
    return d


def differencial2(x, y):
    # Stablish the difference between the X
    diff = np.diff(x)
    c = []
    # Make the value of y into the same of the variable diff
    for i in range(1, len(y)):
        c.append(y[i])
    # Calculate the lineal regression of the values.
    c = Lr.linealregression2(diff, c)
    # Return the lineal regression approximation of the value of y
    d = [0] * len(x)
    for i in range(1, len(diff)):
        d[i] = c[i] + x[i - 1]
    return d


def differencial3(x, y):
    # Stablish the difference between the X
    diff = np.diff(x)
    c = []
    # Make the value of y into the same of the variable diff
    for i in range(1, len(y)):
        c.append(y[i])
    # Calculate the lineal regression of the values.
    c = Lr.linealregression3(diff, c)
    # Return the lineal regression approximation of the value of y
    d = [0] * len(x)
    for i in range(1, len(diff)):
        d[i] = c[i] + x[i - 1]
    return d
