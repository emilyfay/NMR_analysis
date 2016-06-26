# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:31:16 2016

Calculate coefficients for Orthogonal Matching Pursuit model using n features with cross-validation to
select the number of predictors to use in the model
Predict D given selected predictor features (user-specified from available data)
Compare for different cross-validation folds (number of partitions, determines how many
samples are excluded for each CV run)

plot comparison of D and predicted D
calculate R2 value

@author: efay
"""
import feature_load
import matplotlib.pyplot as plt
from sklearn import cluster, preprocessing, linear_model
import numpy as np
import copy

# load features from file
features, feature_names = feature_load.load_features()

#%%

Features = copy.copy(features) # copy before changing
Feature_names = copy.copy(feature_names)
# convert chi to delta chi, differnce from chi of water
# back to 10-6 instead of x 10-6
Features[:,16] = (abs(features[:,16]+9)*0.000001)
# adjust name to show change
Feature_names[16] = 'delta chi'

# choose how between chi, chi^(1/3),and log10(chi) as predictor
# set chi_pick to 'chi', 'chi^1/3' or 'log chi' (loads as chi)
chi_pick = 'chi^1/3'

if chi_pick == 'chi^1/3':
    Features[:,16] = Features[:,16]**0.333
elif chi_pick == 'log chi':
    Features[:,16] = np.log10(Features[:,16])

Feature_names[16] = chi_pick

# make array of predictors from available features
predictors = Features[:,[16,  19, 18, 17, 15]]
predictor_names = Feature_names[[16, 19, 18, 17, 15]]
print('Predictors included :')
for s in predictor_names: print s

M = len(predictor_names)

unscaled_predictors = copy.copy(predictors)

# standardize predictors using the scale function, gives mean 0 and unit variance
SS1 = preprocessing.StandardScaler()
predictors = SS1.fit_transform(predictors) 
scale_pred = SS1.scale_
mean_pred = SS1.mean_ 
#%%

# make array of D values to predict
X = features[:,[2]] # 2 corresponds to logmean D
X= np.log10(X) # take the log to account for range over multiple orders of mag.
X_name = 'log'+feature_names[2].split(' ')[1]
unscaledX = copy.copy(X)

# standardize 
SS2 = preprocessing.StandardScaler()
X = SS2.fit_transform(X)
scale_D = SS2.scale_
mean_D = SS2.mean_

# initialize array for OMP results       
OMP_coefs = np.zeros((M,1))
OMP_predicted = np.zeros((len(predictors),1))
OMP_score = np.zeros(1)

# list of different fold values for cross validation
cv_list = [5,7,10]

# open a file to save calculated coefficients
coefs_file = open('OMP_coefs.txt','w')
coefs_file.write('OMP algorithm with CV to calculate %s' %X_name)
# write the predictor names to file
for pred in predictor_names: coefs_file.write('%s\t' %pred)
# add column for R2 and fold
coefs_file.write('R2\tfold\n')

#make figure to plot predicted vs measured   
plt.figure(figsize=(5*len(cv_list),5))
sb = 1 # subplot index

# loop over the list of cv fold values to calculate the model, plot the results, and save to file
for cv_ in cv_list:
    OMP = linear_model.OrthogonalMatchingPursuitCV(cv = cv_)
    OMPfit = OMP.fit(predictors, X[:,0])
    OMP_coefs = OMPfit.coef_
    OMP_predicted= OMP.predict(predictors)
    OMP_score = OMP.score(predictors, OMP_predicted)
    
    D = X[:,0]*scale_D+mean_D[0]
    Dpredicted = OMP_predicted*scale_D+mean_D[0]
    
    
    # calculate the R2 of the fit
    R2 = 1 - sum((D-Dpredicted)**2)/sum((D-np.mean(D))**2)
    
    print ('CV fold = %d' %cv_) 
    print 'Number of non-zero coefficients:', OMPfit.n_nonzero_coefs_
    print  'R2 :',R2

    for coef in OMP_coefs: coefs_file.write('%f\t' %coef)
    
    coefs_file.write('%f\t%f\n' %(R2, cv_))
    
    plt.subplot(1,len(cv_list),sb)
    plt.scatter(D,Dpredicted)
    plt.plot([min(D),max(D)],[min(D),max(D)])
    plt.axis('tight')
    plt.title('Prediction of %s with CV fold %d' %(X_name, cv_))
    plt.xlabel('measured %s' %X_name)
    plt.ylabel('predicted %s' %X_name)
    
    sb +=1 # increment subplot index

#plt.savefig('Figures/OMPregression_results_%s.eps' %X_name)
coefs_file.close()
