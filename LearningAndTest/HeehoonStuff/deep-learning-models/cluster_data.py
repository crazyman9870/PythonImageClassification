from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist

import numpy as np

import argparse
import csv
#import os

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--labels', required=True, help='path to the csv file with labels')
ap.add_argument('-d', '--data', required=True, help='path to the csv file with data')
ap.add_argument('-r', '--result', required=True, help='path to the csv that will store the dendrogram data')
args = vars(ap.parse_args())

def csv_to_array(filename):
    f = open(filename, 'r')
    reader = csv.reader(f, delimiter=',')
    x = list(reader)
    return x

labels = csv_to_array(args['labels'])
#print(str(labels))
data = csv_to_array(args['data'])
data = np.array(data).astype('float')
#print(data)

print(data.shape)
print('\n')

# generate the linkage matrix
Z = linkage(data, 'ward')

c, coph_dists = cophenet(Z, pdist(data))
print(c)
print('\n')

with open(args['result'], 'w') as f:
    writer = csv.writer(f)
    writer.writerows(Z)
f.close()
