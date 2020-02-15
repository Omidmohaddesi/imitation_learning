from dataset import CrispDataset
from copy import deepcopy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import numpy as np
from dtaidistance import dtw, dtw_visualisation as dtwvis
import seaborn as sns

expert_data_path = "datasets/player_state_actions/"

series = CrispDataset(expert_data_path)
order = series.order.iloc[:, 0:20]

THRESHOLD = 600

trajectories = deepcopy(order)

trajectories1 = trajectories.values[0:22, :]
trajectories2 = trajectories.values[22:46, :]
trajectories3 = trajectories.values[46:68, :]

# s1 = trajectories1[2, :]
# s2 = trajectories1[4, :]

# path = dtw.warping_path(s1, s2)
# dtwvis.plot_warping(s1, s2, path, filename="warp.png")

linked1 = linkage(trajectories1, metric=dtw.distance, method='average', optimal_ordering=True)
linked2 = linkage(trajectories2, metric=dtw.distance, method='average', optimal_ordering=True)
linked3 = linkage(trajectories3, metric=dtw.distance, method='average', optimal_ordering=True)
# linked2 = linkage(trajectories, method='ward')

# clusters = fcluster(linked1, t=300)

labelList1 = np.arange(0, 22)
labelList2 = np.arange(0, 24)
labelList3 = np.arange(0, 22)

# fig, axes = plt.subplots(1, 3, figsize=(3.5, 8))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.tight_layout()
# plt.subplots_adjust(top=0.92, bottom=0.06, hspace=0.3)
plt.subplots_adjust(top=0.9, bottom=0.15, hspace=0.3, left=0.05, right=0.98)

dn1 = dendrogram(linked1,
                 ax=axes[0],
                 orientation='top',
                 labels=labelList1,
                 distance_sort='descending',
                 show_leaf_counts=True,
                 color_threshold=400,
                 leaf_rotation=90,
                 )
axes[0].set_title('Human-Like', fontsize=16)
axes[0].tick_params(labelsize=14)

# fig2 = plt.figure(figsize=(6, 10))
dn3 = dendrogram(linked3,
                 ax=axes[1],
                 orientation='top',
                 labels=labelList3,
                 distance_sort='descending',
                 show_leaf_counts=True,
                 color_threshold=300,
                 leaf_rotation=90,
                 )
axes[1].set_title('OUL without Suggestions', fontsize=16)
axes[1].tick_params(labelsize=14)

dn2 = dendrogram(linked2,
                 ax=axes[2],
                 orientation='top',
                 labels=labelList2,
                 distance_sort='descending',
                 show_leaf_counts=True,
                 color_threshold=250,
                 leaf_rotation=90,
                 )
axes[2].set_title('OUL with Suggestions', fontsize=16)
axes[2].tick_params(labelsize=14)
axes[1].set_xlabel('Player ID', fontsize=16)

# fig.suptitle('Hierarchical Clustering with DTW')

# fig2 = plt.figure(figsize=(10, 7))
# dendrogram(linked2,
#            orientation='top',
#            labels=labelList,
#            distance_sort='descending',
#            show_leaf_counts=True,
#            # color_threshold=THRESHOLD,
#            )
# plt.title('ward')

fig.savefig('player_clusters_h2.png', format='png', dpi=300)
# plt.show()
