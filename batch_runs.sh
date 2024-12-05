#!/bin/bash # COMMENT:The interpreter used to execute the script
# COMMENT: #SBATCH directives that convey submission options:
#SBATCH --job-name=batched_runs
#SBATCH --mail-user=kapnadak@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=10:00
#SBATCH --account=test
#SBATCH --partition=standard
#SBATCH --output=/home/kapnadak/Register_augmented_fine_tuning/slurm_out.log
# COMMENT:The application(s) to execute along with its input arguments and options:
conda 
python QA_script.py 0
python QA_script.py 5
python QA_script.py 10
python QA_script.py 15
python QA_script.py 20
python QA_script.py 25
python QA_script.py 30
python QA_script.py 35
python QA_script.py 40
python QA_script.py 45
python QA_script.py 50
python QA_script.py 55
python QA_script.py 60
python QA_script.py 65
python QA_script.py 70
python QA_script.py 75
python QA_script.py 80
python QA_script.py 85
python QA_script.py 90
python QA_script.py 95
python QA_script.py 100