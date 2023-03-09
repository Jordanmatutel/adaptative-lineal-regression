import decisiontree as dt
import matplotlib.pyplot as plt
import datacreator as dc

dc.data_creator(100)
c = dt.decision_tree()

# Set the data for our graphs.
fig, ax = plt.subplots()

# Graph details
ax.plot(c['Best Result'], color='blue')
ax.scatter(c.index, c['Y'], color='red')
max_y1 = max(c['Best Result'].max(), c['Y'].max())
ax.legend(['Best Result', 'Y'])

plt.show()
