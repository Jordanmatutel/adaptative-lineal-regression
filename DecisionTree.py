import pandas as pd
import linearregression as lr
import derivative as dr
import catboost_model as cbt


def decision_tree():
    # Read the data csv
    d = pd.read_csv("data.csv")
    y = d["y_train"]

    # Create the model using lineal regression.
    linear = lr.linear_regression()
    linear2 = lr.linear_regression2()
    linear3 = lr.linear_regression3()

    # Create the model using derivatives.
    slope = dr.slope_result()

    # Create the model using catboost.
    cb = cbt.catboost_result()

    # Makes one dataframe with the results
    data = pd.DataFrame(columns=["Real Y", "linear1", "linear2", "linear3", "derivative", "catboost"])
    data["Real Y"] = y
    data["linear1"] = linear
    data["linear2"] = linear2
    data["linear3"] = linear3
    data["derivative"] = slope
    data["catboost"] = cb

    # Saves the results for every model
    data.to_csv("All_Results.csv")

    # Takes the columns and calculate the mean error squared.
    mse = ((data - y) ** 2).mean()

    # Return the option with the less mean error.
    lowest_mse = mse["linear1"]
    lowest_col = "linear1"

    for col in mse.index:
        if mse[col] < lowest_mse:
            lowest_mse = mse[col]
            lowest_col = col

    # Return the DataFrame with the selected option with the lowest amount of error
    c = pd.DataFrame(columns=["X", "Y", "Best Result"])
    c["Y"] = y
    c["Best Result"] = data[lowest_col]
    return c
