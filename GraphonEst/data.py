'''

Create graphs from real-world data.
@author: Benjamin Sischka

'''
import os
import numpy as np
import networkx as nx
import scipy.io
import csv
from GraphonEst.graph import ExtGraph

def GraphFromData(data_, dir_, estByDegree, addLabels = True):
    node_labels = None
    ##### Facebook
    if data_ == 'facebook':
        adjMat = nx.to_numpy_array(nx.read_edgelist(os.path.join(dir_, 'Data/facebook') + '/0.edges'))
        # adjMat = nx.to_numpy_array(nx.read_edgelist(os.path.realpath('Data/facebook') + '/0.edges'))  # '../Data/facebook'  # ***
    ##### Human Brain Data
    if data_ == 'brain':
        weightMat = scipy.io.loadmat(os.path.join(dir_, 'Data/human_brain') + '/Coactivation_matrix.mat')['Coactivation_matrix']
        # weightMat = scipy.io.loadmat(os.path.realpath('Data/human_brain') + '/Coactivation_matrix.mat')['Coactivation_matrix']  # '../Data/human_brain'  # ***
        adjMat = (weightMat >= 1e-5).astype('int')
        # an edge between two brain regions means that there is at least one task at which they are coactivated
    ##### Military Alliances
    if data_ == 'alliances':
        adjMat = np.zeros((0, 257)).astype('int')
        with open(os.path.join(dir_, 'Data/alliances') + '/alliances_strong_post_adjMat_2016.csv') as f_cont:
        # with open(os.path.realpath('Data/alliances') + '/alliances_strong_post_adjMat_2016.csv') as f_cont:  # '../Data/alliances'  # ***
            reader = csv.reader(f_cont)
            node_labels = next(reader)[1:]
            for line in reader:
                adjMat = np.append(adjMat, [[int(int_i) for int_i in line[1:]]], axis=0)
        margSum_pos = (adjMat.sum(axis=0) > 0)
        adjMat = adjMat[margSum_pos][:, margSum_pos]
        node_labels = np.array(node_labels)[margSum_pos]
        # remove isolated groups and single connected nodes
        all_other = np.logical_not(np.in1d(node_labels, ['China', 'Cuba', 'North Korea', 'Bosnia and Herzegovina', 'Syria']))
        adjMat = adjMat[all_other][:, all_other]
        node_labels = node_labels[all_other]
    #####
    return(ExtGraph(A = adjMat, estByDegree=estByDegree, labels=node_labels if addLabels else None))

