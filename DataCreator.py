import numpy as np
import pandas as pd


def datacreator(n):
    x_train = np.random.rand(n)
    x_test = np.random.rand(n)
    noise_train = np.random.normal(0, 0.3, n)
    noise_test = np.random.normal(0, 0.3, n)

    data = pd.DataFrame(columns=["x_train", "x_test", "noise_train", "noise_test"])
    data["x_train"] = x_train
    data["x_test"] = x_test
    data["noise_train"] = noise_train
    data["noise_test"] = noise_test
    data.to_csv("data.csv", index=False)
    return data
