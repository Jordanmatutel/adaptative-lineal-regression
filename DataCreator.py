import numpy as np
import pandas as pd


# Creates the csv
def data_creator(n):
    # Creates random numbers between 1 and 0.
    x_train = np.random.rand(n)
    x_test = np.random.rand(n)
    noise_train = np.random.normal(0, 0.3, n)
    noise_test = np.random.normal(0, 0.3, n)

    # The slope of the function used in our data.
    a, b = 2, 3
    y_train = a * x_train + b + noise_train
    y_test = a * x_test + b + noise_test

    # Return the dataframe with the data created.
    data = pd.DataFrame(columns=["x_train", "x_test", "y_train", "y_test"])
    data["x_train"] = x_train
    data["x_test"] = x_test
    data["y_train"] = y_train
    data["y_test"] = y_test
    data.to_csv("data.csv", index=False)
    return data
