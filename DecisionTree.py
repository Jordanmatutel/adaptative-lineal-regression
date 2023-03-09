import pandas as pd
import linealregression as lr
import derivative as dr
import catboost_model as cbt


def decision_tree():
    # Read the data csv
    d = pd.read_csv("data.csv")
    y = d["y_train"]

    # Create the model using lineal regression.
    lineal = lr.lineal_regression()
    lineal2 = lr.lineal_regression2()
    lineal3 = lr.lineal_regression3()

    # Create the model using derivatives.
    slope = dr.slope_result()

    # Create the model using catboost.
    cb = cbt.catboost_result()

    # Makes one dataframe with the results
    data = pd.DataFrame(columns=["lineal1", "lineal2", "lineal3", "derivative", "Catboost"])
    data["lineal1"] = lineal
    data["lineal2"] = lineal2
    data["lineal3"] = lineal3
    data["derivative"] = slope
    data["catboost"] = cb

    # Takes the columns and calculate the mean error squared.
    mse = ((data - y) ** 2).mean()

    # Return the option with the less mean error.
    lowest_mse = mse["lineal1"]
    lowest_col = "lineal1"

    for col in mse.index:
        if mse[col] < lowest_mse:
            lowest_mse = mse[col]
            lowest_col = col


    # Return the DataFrame with the selected option with the lowest amount of error
    c = pd.DataFrame(columns=["X", "Y", "Best Result"])
    c["Y"] = y
    c["Best Result"] = data[lowest_col]
    return c
