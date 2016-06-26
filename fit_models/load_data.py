# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:02:24 2016

@author: efay

functions to load matlab modeling results and lab D-T2 data
"""

from os import listdir
from spyderlib.utils.iofuncs import get_matlab_value
import scipy.io as spio
import numpy
from math import atan2
from cmath import exp

#############################################################################
def load_models(model_folder):
    '''
    function to load the results from the Matlab finite element simulator
    showing the simualted D-T2 response
    
    model parameters are read in based on the folder and file names
    
    returns the dictionary 'Models' which contains:
        M_D: a list containing the array showing decay due to D for each model
        M_T2: a list containing the array showing decay due to T2 for each model
        M_tD: a list of the time vector associated with the D decay
        M_tT2: a list of the time vector associated with the T2 decay
        M_params: a list of model parameters (pore size, rho, chi) for each model
    '''
    #check how many subfolders are in the folder
    X = len(listdir(model_folder))
    # initialize a sufficiently large array to store model parameters
    params = numpy.zeros(shape=(X*30,3))
    D_list = []
    T2_list = []
    deltalist = []
    tD = []
    tT2 = []
    count = 0
    
    # load in all models stored in subfolders in the selected directory
    for f in listdir(model_folder):
        if f == '.DS_Store': continue
        # get model pore size and rho parameters from subfolder name        
        a = f.split('_')
        pore = a[1]
        pore_size = float(a[3])
        rho = float(a[5])
        for B in listdir('%s/%s'%(model_folder,f)):
            if B == '.DS_Store': continue
            # get model chi parameter from file name            
            b = B.split('_')
            b2 = b[1].split('.mat')
            chi = float(b2[0])
            # load results from .mat files
            out = spio.loadmat('%s/%s/%s'%(model_folder,f,B))
            for key, value in list(out.items()):
                out[key] = get_matlab_value(value)
            DT2 = out['DT2']
            T2 = out['T2data']
            T2 = T2/max(T2)
            if count==0:
                deltalist = out['deltalist']
                t = out['t_mat']
                
            params[count,0:3] = pore_size,rho,chi
            
            # get the D decay from the first row of DT2
            D_list.append(DT2[:,0])
            # get the T2 decay from the first column of DT2, starting at 10 (2 ms)
            T2_list.append(T2[10:]*DT2[0,1]) #scale by the second echo for smallest delta
            count+=1
            
    
    tD = t[:,:2]
    tT2 = t[0,:]         
    params = params[:count,:]  
    Models = {'M_D': D_list, 'M_T2': T2_list, 'M_tD': tD, 'M_tT2':tT2, 'M_params': params}
    print('Loaded model data from %s for %d models \n' %(model_folder, count))
    return(Models)
 
#############################################################################
def rotate_data(Rdata,Idata):
    '''
    function to rotate real and imaginary components of NMR decay data
    to maximize the amplitude of the real signal
    '''
    Data = Rdata+Idata*1j
    phi = atan2(sum(Idata[0,1:10]),sum(Rdata[0,1:10]))
    Y = exp(-1j*phi)*Data
    Rdata_rot = Y.real
    noise = Y.imag
    
    return Rdata_rot,noise
    
#############################################################################
def load_lab_data(data_folder):
    '''
    function to load in laboratory D-T2 data for multiple samples in a directory
    returns a dictionary with an entry for each sample with:
        deltalist: list of diffusion times
        DT2:  2D matrix of signal amplitudes
        t_mat: 2D matrix of measurement times
    '''
     
    te = 200*1e-6 # define the echo spacing as that used in the experiment
                                        
    d = {} # initialize dictionary to store sample data
    NS = 0 #initialize counter to track how many samples were loaded
        
    for sample in listdir(data_folder):
        sample_name = sample
        delta_list = []
        Data_list = []
        t_list = []
        for run in listdir('%s/%s' %(data_folder,sample)):
            if run == '.DS_Store': continue
            dlist = numpy.loadtxt('%s/%s/%s/Tdata.csv'%(data_folder,sample,run), dtype = float, delimiter = ',')
            raw_data = numpy.loadtxt('%s/%s/%s/Rawdata2d.csv'%(data_folder,sample,run), dtype = float, delimiter = ',')
            dlist = list(dlist[:,0]*1e-6) # convert delta times from ms to s
            # extract real and imaginary components from saved data
            Rdata = raw_data[:,0::2]
            Idata = raw_data[:,1::2]
            # rotate the data to ensure real component is maximized
            Data, noise = rotate_data(Rdata,Idata)
            # create the time matrix based on delta values in dlist and te
            t_mat = numpy.vstack((2*numpy.tile(dlist,(1,1)),4*numpy.tile(dlist,(1,1)),4*numpy.tile(dlist,(1,1))+te))
            delta_list.append(dlist)
            Data_list.append(Data)
            t_list.append(t_mat)
        sample_dict = {'deltalist': delta_list, 'DT2': Data_list, 't_mat': t_list}    
        d[sample] = sample_dict    
        NS += 1
        
    print('Loaded data from %s for %d samples \n' %(data_folder, NS))
    return(d)




