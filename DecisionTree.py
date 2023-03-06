import LinealRegression as lr
import DifferencialRegression as dr
import CatBoost as cb
import DataCreator as dc


def decisiontree():
    # Create the set of data with the amount of 1000 samples.
    data = dc.datacreator(1000)

    # Take the data to use it in our models.
    x_train = data["x_train"]
    noise_test = data["noise_test"]
    noise_train = data["noise_train"]

    # Create the models.
    lineal = lr.linealregression(noise_train, x_train)
    lineal2 = lr.linealregression2(noise_train, x_train)
    lineal3 = lr.linealregression3(noise_train, x_train)
    diff = dr.differencial(noise_train, x_train)
    diff2 = dr.differencial2(noise_train, x_train)
    diff3 = dr.differencial3(noise_train, x_train)

    # Count the mean amount of error in every model.
    linealerror = lr.meanError(noise_test, lineal)
    linealerror2 = lr.meanError(noise_test, lineal2)
    linealerror3 = lr.meanError(noise_test, lineal3)
    diffError = lr.meanError(noise_test, diff)
    diffError2 = lr.meanError(noise_test, diff2)
    diffError3 = lr.meanError(noise_test, diff3)

    # Sort the results
    c = (linealerror, linealerror2, linealerror3, diffError, diffError2, diffError3)

    return min(c)
