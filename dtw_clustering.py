from dataset import CrispDataset
import pandas as pd
import numpy as np
import random
from copy import deepcopy
from scipy import interpolate
from dtaidistance import dtw
import matplotlib.pyplot as plt
from _plotly_future_ import v4_subplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots

'''
The following code is retrieved and adapted from here: 
https://towardsdatascience.com/time-series-hierarchical-clustering-using-dynamic-time-warping-in-python-c8c9edf2fda5
'''

expert_data_path = "datasets/player_state_actions/"

series = CrispDataset(expert_data_path)
order = series.order.iloc[:, 0:20]

THRESHOLD = 300

trajectories = deepcopy(order)
distanceMatrixDictionary = {}

trajectories = trajectories.values
# distanceMatrix = dtw.distance_matrix(trajectories)

trajectoriesSet = {}
for key, value in enumerate(trajectories):
    trajectoriesSet[(str(key),)] = [value]

trajectories = deepcopy(trajectoriesSet)
distanceMatrixDictionary = {}

iteration = 1
while True:
    distanceMatrix = np.empty((len(trajectories), len(trajectories),))
    distanceMatrix[:] = np.nan

    for index1, (filter1, trajectory1) in enumerate(trajectories.items()):
        tempArray = []

        for index2, (filter2, trajectory2) in enumerate(trajectories.items()):

            if index1 > index2:
                continue

            elif index1 == index2:
                continue

            else:
                unionFilter = filter1 + filter2
                sorted(unionFilter)

                if unionFilter in distanceMatrixDictionary.keys():
                    distanceMatrix[index1][index2] = distanceMatrixDictionary.get(unionFilter)

                    continue

                metric = []
                for subItem1 in trajectory1:

                    for subItem2 in trajectory2:
                        metric.append(dtw.distance(subItem1, subItem2, psi=1))

                metric = max(metric)

                distanceMatrix[index1][index2] = metric
                distanceMatrixDictionary[unionFilter] = metric

    minValue = np.min(list(distanceMatrixDictionary.values()))

    if minValue > THRESHOLD:
        break

    minIndices = np.where(distanceMatrix == minValue)
    minIndices = list(zip(minIndices[0], minIndices[1]))

    minIndex = minIndices[0]

    filter1 = list(trajectories.keys())[minIndex[0]]
    filter2 = list(trajectories.keys())[minIndex[1]]

    trajectory1 = trajectories.get(filter1)
    trajectory2 = trajectories.get(filter2)

    unionFilter = filter1 + filter2
    sorted(unionFilter)

    trajectoryGroup = trajectory1 + trajectory2

    trajectories = {key: value for key, value in trajectories.items()
                    if all(value not in unionFilter for value in key)}

    distanceMatrixDictionary = {key: value for key, value in distanceMatrixDictionary.items()
                                if all(value not in unionFilter for value in key)}

    trajectories[unionFilter] = trajectoryGroup

    print(iteration, 'finished!')
    iteration += 1

    if len(list(trajectories.keys())) == 1:
        break

for key, _ in trajectories.items():
    print(key)

for key, value in trajectories.items():

    if len(key) == 1:
        continue

    figure = make_subplots(rows=1, cols=1)
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(value))]

    for index, subValue in enumerate(value):
        figure.add_trace(go.Scatter(x=list(range(0, len(subValue))), y=subValue,
                                    mode='lines', marker_color=colors[index], line=dict(width=4), line_shape='spline'),
                         row=1, col=1,
                         )

        '''oldScale = np.arange(0, len(subValue))
        interpolateFunction = interpolate.interp1d(oldScale, subValue)

        newScale = np.linspace(0, len(subValue) - 1, MAX_LEN_OF_TRAJECTORY)
        subValue = interpolateFunction(newScale)

        figure.add_trace(go.Scatter(x=list(range(0, len(subValue))), y=subValue,
                                    mode='lines', marker_color=colors[index]), row=1, col=2)'''

    figure.update_layout(showlegend=False, height=600, width=900)
    figure.show()

