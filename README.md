# SNOWstorm

ML-based emulator model for downscaling of near-surface winds and wind-driven redistribution of snow in mountain environments.
The model was trained on data from semi-idealized WRF simulations. It can be driven currently with atmospheric input from ERA5 and WRF, coupling to other input data to come.


## Repository Structure 

snowstorm.py
Main script to run SNOWstorm

./snowstorm_helpers
contains helper scripts data curation etc.

./model
contains final models plus helping files (normfactors, namelists..)

./train
contains scripts used during model training
- nn_wrf_main, nn_wrf_train, nn_wrf_namelist, nn_wrf_helpers, nn_wrf_diag, nn_wrf_crossvalid

./data
contains training data used for SNOWstorm

 
