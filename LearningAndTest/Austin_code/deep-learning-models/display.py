# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--data', required=True, help='path to the csv file with data')
args = vars(ap.parse_args())

Z = np.genfromtxt(args['data'], delimiter=',')

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()