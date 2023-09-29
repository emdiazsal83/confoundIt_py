# confoundIt_py
python "proxy confounder factorization gradient descent (PCF-GD)" code
PCF-GD method described in manuscript "Recovering latent confounders from high-dimensional
proxy variables"

The src folder includes the following files:
- funcs_LNC.py: functions to run non-linear version of PCF-GD
- funcs_LNC_lin.py: functions to run linear version of PCF-GD
- processResults.py: helper functions to setup parallel slurm runs and to read in results there-of
- experiment.py: main program to run PCF-GD on one dataset and set of parameters
- slurm_script.py: main program to be called by LNC_job.sh batch script which furn experiment.py for a dataset and set of hyperparameters indexed by job
- LNC.job.sh: slurm script to call slurm_script for 1-N jobs

The notebooks folder includes the following files:
- LN_4_sep_confounders_causeEffect.ipynb which runs non-linear version of PCF-GD and plots evolution of optimization according to several metrics
- LN_4_sep_confounders_causeEffect_linearize.ipynb which runs linear version of PCF-GD and plots evolution of optimization according to several metrics
- explore_results.ipynb which reads in results for a batch-run and plots figures for performance
