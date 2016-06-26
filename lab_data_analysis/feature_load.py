# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:24:23 2016

for loading in the feature data for clustering and regression
@author: efay
"""
import csv, copy
import numpy as np

def load_features():
    '''
    function to load in the feature data from files in the current folder
    
    returns a list of the feature names and an array of the features
    '''
    # Load in data from file
    filename1 = 'D_params.txt' #file containing measured D parameters w/ headers
    filename2 = 'G_params.txt' #file containing measured G parameters w/ headers
    filename3 = 'Sample_lab_data.csv' #file containing additional lab data w/ headers
    
    # open files for reading, saving the first line of header data
    reader1 = csv.reader(open(filename1, 'rU'), delimiter=',')
    headers1 = reader1.next()
    reader2 = csv.reader(open(filename2, 'rU'), delimiter=',')
    headers2 = reader2.next()
    reader3 = csv.reader(open(filename3, 'rU'), delimiter=',')
    headers3 = reader3.next()
    
    # get headers for columns to use in analysis
    headers1 = headers1[3:11]
    headers2 = headers2[1:8]
    headers3 = headers3[2:10]
    
    # read in sample names from first line of files
    sample_list1 = [x[0] for x in reader1] 
    sample_list2 = [x[0] for x in reader2]  
    sample_list3 = [x[0] for x in reader3]
    
    # get data to use in analysis
    D_stats = np.loadtxt(filename1, dtype = float, delimiter = ',', usecols = range(3,11), skiprows = 1)
    G_stats = np.loadtxt(filename2, dtype = float, delimiter = ',', usecols = range(1,8), skiprows = 1)
    Sample_properties = np.loadtxt(filename3, dtype = float, delimiter = ',', usecols = range(2,10), skiprows = 1)
    
    # need to sort by sample name since ordering if different in the input files
    # zip sample names with sample data
    D1 = zip(sample_list1,D_stats)
    G1 = zip(sample_list2, G_stats)
    S1 = zip(sample_list3, Sample_properties)
    #sort by sample name
    D1.sort(key = lambda t: t[0])
    G1.sort(key = lambda t: t[0])
    S1.sort(key = lambda t: t[0])
    
    # unzip sorted data
    samples1, D_stats = zip(*D1)
    samples1 = list(samples1)
    D_stats = np.array(D_stats)
    samples2, G_stats = zip(*G1)
    samples2 = list(samples2)
    G_stats = np.array(G_stats)
    samples3, Sample_properties = zip(*S1)
    samples3 = list(samples3)
    Sample_properties = np.array(Sample_properties)
    
    # combine the headers from the 3 files into one array
    feature_names = np.hstack((headers1, headers2, headers3))
    
    # combine the features from the 3 files into one array
    features = np.hstack((D_stats, G_stats, Sample_properties))
    
    print('Loaded features from file.\n')
    return(features, feature_names)

