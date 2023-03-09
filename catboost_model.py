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
    prediction_y = []
    for x in x_test:
        y = model.predict([x])
        prediction_y.append(y)

    return prediction_y
