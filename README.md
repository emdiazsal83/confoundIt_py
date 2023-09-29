# confoundIt_py
python proxy confounder factorization gradient descent (PCF-GD) code

The src folder includes the following files:
- funcs_LNC.py: functions to run non-linear version of PCF-GD
- funcs_LNC_lin.py: functions to run linear version of PCF-GD
- processResults.py: helper functions to setup parallel slurm runs and to read in results there-of
- experiment.py: main program to run PCF-GD on one dataset and set of parameters
- slurm_script.py: main program to be called by LNC_job.sh batch script which furn experiment.py for a dataset and set of hyperparameters indexed by job
- LNC.job.sh: slurm script to call slurm_script for 1-N jobs
