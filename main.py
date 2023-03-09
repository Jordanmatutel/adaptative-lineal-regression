import decisiontree as dt
import matplotlib.pyplot as plt
import datacreator as dc
import pandas as pd

# Creates the data and use the decision tree
dc.data_creator(100)
r = dt.decision_tree()

# Saves the data and shows the best result
c = pd.read_csv("data.csv")
result = pd.DataFrame(columns=["X", "Y", "Best Result"])
result["X"] = c["x_test"]
result["Y"] = c["y_train"]
result["Best Result"] = r["Best Result"]
result.to_csv("result.csv")

# Set the data for our graphs.
fig, ax = plt.subplots()

# Graph details
ax.plot(r['Best Result'], color='blue')
ax.scatter(r.index, r['Y'], color='red')
max_y1 = max(r['Best Result'].max(), r['Y'].max())
ax.legend(['Best Result', 'Y'])

plt.show()
