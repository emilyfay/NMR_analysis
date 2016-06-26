# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:31:16 2016

Code to run cluster analysis on laboratory data for predicting diffusion and internal gradients
Calculates clusters using 3 algorithms:
DBSCAN
Affinity Propagation
Agglomerative Clustering

write the cluster labels to file, indicating the parameters used

@author: efay
written in Spyder IDE
"""
#%%
import feature_load
import matplotlib.pyplot as plt
from sklearn import cluster, preprocessing, linear_model
import numpy as np
import csv, copy

#%%
# load in the feature data from file
features, feature_names = feature_load.load_features()
 
#%%
##########################################################################
# Cluster Analysis
##########################################################################
    
# select features to include in cluster analysis 
cluster_features = features[:,[0,2,6,10,13,14,22,16,19,18,17,21,15]]
cluster_feature_names = feature_names[[0,2,6,10,13,14,22,16,19,18,17,21,15]]

# list of features to take logarithm of for analysis
loglist = [0,1,3,6]

for nl in loglist:
    cluster_features[:,nl] = np.log10(cluster_features[:,nl])
    cluster_feature_names[nl] = 'Log '+cluster_feature_names[nl]

# make a copy of features prior to scaling
unscaled_cluster_features = copy.copy(cluster_features) 

# standardize predictors using the scale function for mean zero and unity variance
for n in range(len(cluster_features[1,:])):
    cluster_features[:,n] = preprocessing.scale(cluster_features[:,n]) 
#%%
##############################################################################    
# DBSCAN clustering algorithm
# views clusters as high density areas separated by areas of low density, can be any shape
# min_samples parameter determines how many samples are required to form a cluster
# eps parameter defines max distance between the core cluster point and points in cluster
    
eps_list = [0.5,1,2]
min_list = [2,3,5] #small dataset, so allowing smaller clusters

# file for writing in cluster results
DBcluster_file = open('DBSCAN_cluster_labels.txt','w')
# file for saving parameters used to calculate results
DBparams_file = open('DBSCAN_parameters.txt','w')

DBparams_file.write('Parameters used to calculate clusters in DBSCAN_cluster_labels.txt \n')
DBparams_file.write('eps, min_samples \n')

for min_samples_ in min_list:
    for eps_ in eps_list:
        # calculate clusters in the data for the given eps and min-sample parameters
        db = cluster.DBSCAN(eps=eps_, min_samples=min_samples_).fit(cluster_features)
        labels_db = db.labels_
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_DB = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        print('Number of clusters from DBSCAN with eps=%d and %d min samples: %d' %(eps_,min_samples_, n_clusters_DB))
       
        # write cluster results to file
        for label in labels_db:
            DBcluster_file.write('%d,' %label)
        DBcluster_file.write('\n')
        # write cluster parameters to file
        DBparams_file.write('%d, %d \n' %(eps_, min_samples_))

DBcluster_file.close() 
DBparams_file.close()     
#%%
##############################################################################
# Affinity propagation clustering algorithm
# creates clusters by sending messages between pairs of samples until convergence
# chooses "exemplar points"
# damping parameter, between 0.5 and 1 

# choose list of damping parameters to compare
damping_list = [0.75, 0.85, 0.9]


# file for writing in cluster results
APcluster_file = open('AP_cluster_labels.txt','w')
# file for saving parameters used to calculate results
APparams_file = open('AP_parameters.txt','w')

APparams_file.write('Parameters used to calculate clusters in AP_cluster_labels.txt \n')
APparams_file.write('damping factor \n')

for damping_ in damping_list:
        # calculate clusters in the data for the given eps and min-sample parameters
        ap = cluster.AffinityPropagation(damping = damping_).fit(cluster_features)
        labels_ap = ap.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_AP = len(set(labels_ap)) - (1 if -1 in labels_ap else 0)
        print('Number of clusters from AP with damping = %0.2f : %d' %(damping_, n_clusters_AP))
       
        # write cluster results to file
        for label in labels_ap:
            APcluster_file.write('%d,' %label)
        APcluster_file.write('\n')
        # write cluster parameters to file
        APparams_file.write('%d \n' %(damping_))

APcluster_file.close() 
APparams_file.close()    
#%%
##############################################################################
# Agglomerative Clustering algorithm
# performs hierarchical clustering using a bottom-up approach, 
# merging clusters following a variance-minimization approach
# parameter n_clusters sets how many clusters to solve for

n_list = [2,3,4,5]

# file for writing in cluster results
AGcluster_file = open('AG_cluster_labels.txt','w')
# file for saving parameters used to calculate results
AGparams_file = open('AG_parameters.txt','w')

AGparams_file.write('Parameters used to calculate clusters in AG_cluster_labels.txt \n')
AGparams_file.write('number of clusters \n')

for n_clusters_ in n_list:
    ag = cluster.AgglomerativeClustering(n_clusters = n_clusters_).fit(cluster_features)
    labels_ag = ag.labels_
    
    # Number of clusters in labels (specified by user)
    n_clusters_AG = len(set(labels_ag)) - (1 if -1 in labels_ag else 0)
    
    print('Number of clusters used for Agglomerative Clustering: %d' % n_clusters_AG)
    
    # write cluster results to file
    for label in labels_ag:
        AGcluster_file.write('%d,' %label)
    AGcluster_file.write('\n')
    # write cluster parameters to file
    AGparams_file.write('%d\n' %(n_clusters_))
    
AGcluster_file.close() 
AGparams_file.close()    

#%%
##############################################################################

features_file = open('clustering_features.txt','w')

for c_name in cluster_feature_names:
    features_file.write('%s, ' %c_name)

features_file.write('\n')

for n in range(len(labels_ag)):
    for m in range(len(cluster_feature_names)):
        features_file.write(('%2.3g, ' %cluster_features[n,m]))
    features_file.write('\n')
    
features_file.close()

print('\n Saved cluster analysis results.')

'''
Note:
Cluster analysis results were loaded in to Matlab and used to group the samples (by color) on a parallel coordinates plot.
The parallel coordinate plots were used to visualize the clusters given the 13 different measured parameters.
The "value" of the output from each clustering run was determined qualitatively from the parallel coordinate plots,
based on prior knowledge of the samples, and the ability of the clusters to separate between samples with differences
in the parameters we wanted to predict (the first three columns in the features list).
'''