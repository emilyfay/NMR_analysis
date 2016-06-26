# NMR_analysis
Codes for analyzing laboratory data for sediment samples and comparing measured NMR data to simulated data
Supports research presented in Fay, E. (2016). Nuclear Magnetic Resonance Diffusion Measurements for Characterization of the Near-Surface. PhD thesis, Stanford University. 

Contains two folders:
- lab_data_analysis : analyze laboratory data for sediment samples using clustering and regression algorithms
- fit_models : compare measured signal decays to simulated deacys and select models with the best fit to the lab data

lab_data_analysis contents:
-Cluster_analysis_lab_data.py: 
      Code to run cluster analysis on laboratory data for predicting diffusion and internal gradients
      Calculates clusters using 3 algorithms:
      DBSCAN
      Affinity Propagation
      Agglomerative Clustering
      write the cluster labels to file, indicating the parameters used
-D_predictor.py:
      Calculate coefficients for Orthogonal Matching Pursuit model using n features with cross-validation to
      select the number of predictors to use in the model
      Predict D given selected predictor features (user-specified from available data)
      Compare for different cross-validation folds (number of partitions, determines how many
      samples are excluded for each CV run)
      plot comparison of D and predicted D
      calculate R2 value
      write the coefficients to file
- feature_load.py:
      script containing the function loead_features called by the scripts listed above to load in the feature data

- D_params.txt:
    text file containing features related to D, the diffusion coefficient, for 34 sediment samples
- G_params.txt:
    text file containing features related to G, the measured internal gradients, for 34 sediment samples
- Sample_lab_data.csv:
    file containing data describing laboratory samples, including magnetic susceptibility, iron content, and various characterizations of the NMR response



