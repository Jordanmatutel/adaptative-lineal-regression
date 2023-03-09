import pandas as pd
from catboost import CatBoostRegressor


# Catboost model
def catboost_result():
    # Load the dataset
    data = pd.read_csv("data.csv")
    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]

    # Creates the model
    model = CatBoostRegressor(iterations=50, learning_rate=0.1, loss_function='RMSE')
    model.fit(x_train, y_train, silent=True)
    Y_pred = []
    for x in x_test:
        y_pred = model.predict([x])
        Y_pred.append(y_pred)

    return Y_pred
