# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:34:06 2016

@author: efay

code to load in laboratory D-T2 data for samples in a directory, as well as
numerical modeling results from a different directory, then select models that
are a good fit to the data based on the RMS error in D and T2

plots the models with the best fit as well as their parameters to assess robustness
of the method for predicting properties of lab sample from modeling results

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from math import sqrt
import load_data

# load the laboratory D-T2 data
lab_data = load_data.load_lab_data('lab_data')
# load the numerical modeling results
Models = load_data.load_models('models')

# initialize dictionaries for errors and new sample data 
Errors_dict = {}
Sample_data = {}
#%% 
'''
module to calculate RMS error between model data and laboratory data

for each sample, loops through all models and calculates the error 
for the D component, the T2 component, and for both combined

data is normalized by the maximum amplitude prior to comparison, so only
the shape of the decay curves are compared
''' 
# choose which T2 points to consider for error calculation    
T2points = range(1,15,1)+range(15,100,5)+ range(100,500,20) + range(500,2000,50)

# load in the data from the models
M_D = Models['M_D']
M_tD = Models['M_tD'][:,0]
M_tT2 = Models['M_tT2'][T2points]

s_num = 1

for sample in lab_data:
    
    D_list = []
    tD_list = []
    T2_list = []
    tT2_list = []   
    D_error = []    
    T2_error = []    
    tot_error = []     
    T2models = []
    Dmodels = []
    
    # get lab DT2 and t data for this sample
    dt2 = lab_data[sample]['DT2']
    t = lab_data[sample]['t_mat']
    
    # find maximum of first row for normalizing data
    Dmax = max(dt2[0][:,0])
    
    for n in range(len(dt2)):
        Dlist = dt2[n][:,0]/Dmax # normalize
        D_list.append(Dlist)
        tD_list.append(t[n][0,:])
        

            
    for m in T2points:
        T2_list.append(dt2[0][0,m]/Dmax) # normalize
        tT2_list.append(t[0][1,0]+(m-1)*0.0002) # build time vector
   
    #  convert lists to arrays 
    lab_D = np.hstack(D_list)
    lab_tD = np.hstack(tD_list)
    lab_T2 = np.hstack(T2_list)
    lab_tT2 = np.hstack(tT2_list)
        

    # for every model in M_D, calculate the RMS error with lab D and T2        
    for m in range(len(M_D)):
        
        # normalize by the max signal amplitude
        maxD = (M_D[m][0])
        M_D1 = M_D[m][:]/maxD
        M_T2 = Models['M_T2'][m][T2points,0]/maxD
                
        # interpolate to the same time points as the lab data
        f1 = interpolate.interp1d(M_tD,M_D1)
        M_D_int = f1(lab_tD)
        
        # calcualte the difference between data and model D and T2 decays
        E_D = abs(M_D_int-lab_D)
        E_T2 = abs(M_T2-lab_T2)
        RMSE_D = sqrt((sum(E_D**2)/len(lab_D)))
        RMSE_T2 = sqrt((sum(E_T2**2)/len(lab_T2)))   
        # weight RMSE_D and RMSE_T2 equally in total error
        RMSE_tot = sqrt((sum(E_D**2)/len(lab_D))+(sum(E_T2**2)/len(lab_T2))/2)
        
        D_error.append(RMSE_D)
        T2_error.append(RMSE_T2)
        tot_error.append(RMSE_tot)
        
        # save the normalized models, but only do this for the first sample
        if s_num == 1:      
            T2models.append(M_T2)
            Dmodels.append(M_D1)
    if s_num == 1:
            out_models = {'T2models': T2models, 'Dmodels': Dmodels, 'tD' : M_tD, 'tT2': M_tT2}        
    s_num = 0
            
    Errors_dict[sample] = {'D': D_error,'T2': T2_error, 'Total': tot_error}
    Sample_data[sample] = {'D Amplitudes': lab_D,  'D times': lab_tD, 'T2 Amplitudes': lab_T2, 'T2 times': lab_tT2 }

#%%
'''
selection of models with the best fit according to the selected error type
choose between D errors, T2 errors, or both
makes a list of all the models with errors within a threshold value of the
minimum error

includes a plotting module
'''
    
# parameters ordered pore size, rho, chi
pore_size = Models['M_params'][:,0]
rho = Models['M_params'][:,1]
chi = Models['M_params'][:,2]

# select which error to use to judge model fit
# set to either 'D', 'T2', or 'Total'
error_type = 'Total'

# set the error threshold, a multiplier on the minimum error
# that determines the range of errors to show
error_thresh = 1.1

for sample in Errors_dict:
 
    Errors = Errors_dict[sample][error_type]

    L = len(Errors)
    chi_min = np.zeros(L)
    rho_min = np.zeros(L)
    ps_min = np.zeros(L)
    er_min = np.zeros(L)
    best_fit_D = np.zeros((len(M_tD),L))
    best_fit_T2 = np.zeros((len(M_tT2),L))
    color_ = [] # color scale based on error
    countD = 0
    
    # select the models with error below the threshold
    Min_err = min(Errors)

    for n in range(L):
        if Errors[n] < Min_err*error_thresh:
            chi_min[countD] = chi[n]
            rho_min[countD] = rho[n]
            ps_min[countD] = pore_size[n]
            er_min[countD] = Errors[n]
            best_fit_D[:,countD] = out_models['Dmodels'][n]
            best_fit_T2[:,countD] = out_models['T2models'][n]
            color_.append(Errors[n])
            countD+=1
        if Errors[n] == Min_err:
            chi_fit = chi[n]
            rho_fit = rho[n]
            ps_fit = pore_size[n]
            pred_D = out_models['Dmodels'][n]
            pred_T2 = out_models['T2models'][n]
    
    # trim arrays to number of non-zero entries
    chi_min = chi_min[:countD]
    rho_min = rho_min[:countD]
    ps_min = ps_min[:countD]
    er_min = er_min[:countD]
    D_best_fit_D = best_fit_D[:,:countD]
    D_best_fit_T2 = best_fit_T2[:,:countD]
    
    print ('Best fit parameters for sample %s based on %s error: ' %(sample, error_type))
    print 'pore size',ps_fit, 'chi', chi_fit,  'rho', rho_fit, '\n'
     
 
    '''
    plotting module
     
    for each sample, a figure is produced to show:
    - the model parameters for the best fit models, color shows the relative error
    compared to the minimum error, marked with an 'x'
    - the comparison of the selected models to the lab D decay with the best fit model
    shown in red, and other models in grey with color indicating error
    - the comparison of the selected models to the lab T2 decay with the best fit model
    shown in red, and other models in grey with color indicating 
    '''
    sample_DX = Sample_data[sample]['D times']
    sample_DY = Sample_data[sample]['D Amplitudes']
    sample_T2X = Sample_data[sample]['T2 times']
    sample_T2Y = Sample_data[sample]['T2 Amplitudes']
    
    # make a colormap based on the errors
    cmap = plt.get_cmap('gray')
    colors = [cmap(i) for i in color_]
    
    plt.figure(figsize=(14,5))
    fig = plt.gcf()
    
    ax = fig.add_subplot(131, projection='3d')
    sc = ax.scatter(np.log10(chi_min),np.log10(rho_min),np.log10(ps_min), s = 50, c=color_, lw = 0.2, alpha = 1)
    ax.scatter(np.log10(chi_fit),np.log10(rho_fit),np.log10(ps_fit), s = 70, marker='x', c='r', lw = 2, alpha = 1)
         
    ax.set_xlabel('chi')
    ax.set_ylabel('rho')
    ax.set_zlabel('pore size')
    ax.set_xlim(-6, -2)
    ax.set_ylim(-6, -1.5)
    ax.set_zlim(-6, -3.5)
    plt.title('%s error, Sample %s' %(error_type, sample))
    plt.colorbar(sc)
    plt.subplot(132)
     
    for n in range(countD):
        plt.plot(M_tD,best_fit_D[:,n],lw = 0.5, color = colors[n])
    plt.plot(M_tD,pred_D, color = 'b', lw = 0.7)
    
    plt.scatter(sample_DX, sample_DY, lw = 0.2, color = 'k', s = 6)
    plt.axis([min(sample_DX),max(sample_DX),0,1])
    plt.title('Fit to D decay')
    
    plt.subplot(133)
    
    for n in range(countD):
        plt.plot(M_tT2,best_fit_T2[:,n],  lw = 0.5, color = colors[n])
    plt.plot(M_tT2,pred_T2,color = 'r', lw = 0.7)
    plt.scatter(sample_T2X, sample_T2Y, lw = 0.1, color = 'k', s  = 6)
    plt.axis([min(sample_T2X),max(sample_T2X),0,1])
    plt.title('Fit to T2 decay')
    
    #plt.savefig('Figures/Model_fit_%s_%s.eps' %(error_type, sample)
