#!/bin/bash
#SBATCH --account=rpp-kshook
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-7:00:00
#SBATCH --mem=48G
#SBATCH --job-name=esp_analysis
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --output=slurm_logs/slurm_%j.out

module restore scimods
source ~/virtual-envs/scienv/bin/activate

initialization_date='04-01'
subbasin_id= '58232'

python 01_generate_ensemble.py "$initialization_date" "$subbasin_id"

python 02_run_ensemble.py "$subbasin_id"






