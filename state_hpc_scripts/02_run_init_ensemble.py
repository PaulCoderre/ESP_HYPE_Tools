#!/usr/bin/env python
# coding: utf-8

# ## Description
# __________
# 
# This script reads the .nc forcing files and runs HYPE for each year and ensemble member.
# It automatically loads initial states from pre-saved state files in the model directory.
# State files must be named state_saveYYYYMMDD.txt and match the forecast start date.

# ### Import Libraries

import os
import re
import pandas as pd
from datetime import datetime
import xarray as xr
import numpy as np
import shutil
import subprocess
import time
import sys

# ### Inputs

# esp ensemble decisions
subbasin_ids = ['58208', '58213', '58223', '58408', '58643', '58308', '58346', '58425', '58356', '58363', '58418', '58290', '58328', '58398', '58242'] 

hype_command = './hype'     # HYPE executable command

esp_forcing_directory = './esp_init_forcings/apr1_start_sep30_end_no_spinup/'  # directory with ESP forcings (no spin-up)
model_directory = '../state_model/'    # directory containing model files AND state files
output_directory = './esp_init_results/apr1_start_sep30_end/'         # output location
runs_per_script = 1

# ### Run Ensemble

# Access command line argument for run number
if len(sys.argv) != 2:
    print("Usage: python run_esp.py <run_number>")
    sys.exit(1)

run_number = int(sys.argv[1])
print(f'Run number = {run_number}')

# Record start time
start_time = time.time()

# Save current directory
current_directory = os.getcwd()

# Define the working directory
working_directory = f"{run_number}_esp_working_directory"
os.makedirs(working_directory, exist_ok=True)

# Dictionary to hold paths for each subbasin
subbasin_output_dirs = {}

# Create a results directory for each subbasin
for sbid in subbasin_ids:
    output_file = f"{sbid}_esp_results"
    full_output_path = os.path.join(output_directory, output_file)
    os.makedirs(full_output_path, exist_ok=True)
    subbasin_output_dirs[sbid] = full_output_path

# Get list of directories containing forcings for each esp year
esp_directories = [name for name in os.listdir(esp_forcing_directory) 
                   if os.path.isdir(os.path.join(esp_forcing_directory, name)) 
                   and not name.startswith('.')]

# Copy model files into working directory (excluding state files for now)
model_contents = os.listdir(model_directory)
for item in model_contents:
    # Skip state files - we'll copy the correct one later
    if item.startswith('state_save') and item.endswith('.txt'):
        continue
    
    source = os.path.join(model_directory, item)
    destination = os.path.join(working_directory, item)
    if not os.path.exists(destination):
        if os.path.isdir(source):
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

# Define ranges for parallel processing
low_range = run_number * runs_per_script - runs_per_script
upper_range = run_number * runs_per_script

print(f'Processing directories from {low_range} to {upper_range}')

# Slice the directory list based on the specified range
directory_subset = esp_directories[low_range:upper_range]
print(f"Processing: {directory_subset}")

# Iterate over each year of ESP
for forcing_directory in directory_subset:
    
    print(f"\n{'='*60}")
    print(f"Processing forcing directory: {forcing_directory}")
    print(f"{'='*60}")
    
    # Get list of each forcing .nc file in the directory
    file_names = os.listdir(os.path.join(esp_forcing_directory, forcing_directory))
    file_names = [file for file in file_names if file.endswith('.nc')]

    # Initialize dictionary of forcing files
    forcings = {}
    
    # Read each .nc forcing file
    for file_name in file_names:
        try:
            dataset = xr.open_dataset(os.path.join(esp_forcing_directory, forcing_directory, file_name))
            dataset_name = os.path.splitext(file_name)[0]
            forcings[dataset_name] = dataset
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            
    # Extract year and dates from forcing file
    esp_date = forcings['Pobs'].time.values[-1]
    esp_date_str = str(esp_date)
    esp_year = esp_date_str.split("-")[0]  # Extract year for filename
    
    # Extract first and last date from forcing file
    new_bdate_value = pd.to_datetime(str(forcings['Pobs'].time.values[0])).strftime('%Y-%m-%d')
    new_edate_value = pd.to_datetime(str(forcings['Pobs'].time.values[-1])).strftime('%Y-%m-%d')
    
    print(f"Forecast period: {new_bdate_value} to {new_edate_value}")

    # Define state file name based on start date
    state_date = new_bdate_value.replace('-', '')  # Format: YYYYMMDD
    state_filename = f'state_save{state_date}.txt'
    state_source = os.path.join(model_directory, state_filename)
    
    # Check if state file exists
    if not os.path.exists(state_source):
        print(f"ERROR: State file not found: {state_source}")
        print(f"Please ensure you have generated state files for all forecast initialization dates.")
        continue
    
    # Copy state file to working directory
    state_destination = os.path.join(working_directory, state_filename)
    shutil.copy2(state_source, state_destination)
    print(f"Copied state file: {state_filename}")

    # Update info.txt with dates and enable state loading
    with open(f'./{working_directory}/info.txt', 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        
        # Track whether we've set instate
        instate_found = False
        
        for line in lines:
            if line.startswith('bdate'):
                f.write(f'bdate {new_bdate_value}\n')
            elif line.startswith('edate'):
                f.write(f'edate {new_edate_value}\n')
            elif line.startswith('cdate'):
                f.write(f'cdate {new_bdate_value}\n')
            elif line.startswith('instate'):
                f.write('instate Y\n')  # Enable state loading
                instate_found = True
            else:
                f.write(line)
        
        # If instate wasn't in the file, add it
        if not instate_found:
            f.write('\ninstate Y\n')
        
        f.truncate()
    
    print(f"Updated info.txt: bdate={new_bdate_value}, edate={new_edate_value}, instate=Y")

    # Find the number of ensemble members from the forcing files
    num_ensemble_members = forcings['Pobs'].sizes['ensemble_member']
    print(f"Number of ensemble members: {num_ensemble_members}")
    
    # Uncomment for debugging (only run 3 members)
    # num_ensemble_members = 3

    # Initialize results list for each subbasin
    subbasin_output_lists = {sbid: [] for sbid in subbasin_ids}
    
    # Loop through each ensemble member
    for ensemble_member_num in range(num_ensemble_members):
        
        print(f"\n  Processing ensemble member {ensemble_member_num + 1}/{num_ensemble_members}")

        # Write forcing files for this ensemble member
        for dataset_name, dataset in forcings.items():
            try:
                # Extract data for given ensemble member
                data = dataset.sel(ensemble_member=ensemble_member_num)
                data = data.to_dataframe()
                
                # Drop 'ensemble_member' column 
                data = data.drop('ensemble_member', axis=1)
                
                # Pivot table to have time in index and subbasin as headers
                data = data.pivot_table(index='time', columns='subbasin')
                
                # Remove the word "subbasin" from the header label
                data.columns = pd.MultiIndex.from_tuples(
                    [(col[0], col[1]) if col[0] != 'subbasin' else (None, col[1]) 
                     for col in data.columns]
                )
                
                # Save DataFrame to tab-separated text file
                output_file = dataset_name + '.txt'
                data.to_csv(f'{working_directory}/{output_file}', 
                           sep='\t', header=True, index=True, index_label=False)
                
                # Remove the extra line above the header
                with open(f'{working_directory}/{output_file}', 'r') as f:
                    lines = f.readlines()
                with open(f'{working_directory}/{output_file}', 'w') as f:
                    f.writelines(lines[1:])
                
                # Add index label to files
                with open(f'{working_directory}/{output_file}', 'r') as file:
                    lines = file.readlines()
                lines[0] = 'date ' + lines[0]
                with open(f'{working_directory}/{output_file}', 'w') as file:
                    file.writelines(lines)
                    
            except Exception as e:
                print(f"    Error processing dataset {dataset_name}: {e}")

        # Run HYPE
        result = subprocess.run(hype_command, shell=True, cwd=working_directory)
        
        if result.returncode != 0:
            print(f"    ERROR: HYPE execution failed. Return code: {result.returncode}")
            continue
            
        # Collect outputs for each subbasin
        for sbid in subbasin_ids:
            try:
                output_path = os.path.join(working_directory, f'00{sbid}.txt')
                results = pd.read_csv(output_path, sep='\t', index_col=0)
                results = results.iloc[1:]  # Remove duplicate header line if present
                
                # Convert to xarray Dataset
                ds = results.to_xarray()
                ds = ds.expand_dims(ensemble_member=[ensemble_member_num])
                
                subbasin_output_lists[sbid].append(ds)
                
            except Exception as e:
                print(f"    Error reading output for subbasin {sbid}: {e}")
    
    # Concatenate and save results for each subbasin
    for sbid in subbasin_ids:
        try:
            final_ds = xr.concat(subbasin_output_lists[sbid], dim='ensemble_member')
            subbasin_dir = subbasin_output_dirs[sbid]
            output_nc_path = os.path.join(subbasin_dir, f'{esp_year}_esp.nc')
            final_ds.to_netcdf(output_nc_path)
            print(f'\nSaved NetCDF for subbasin {sbid}: {output_nc_path}')
        except Exception as e:
            print(f"Error saving NetCDF for subbasin {sbid}: {e}")
    
    print(f'\n{esp_year} completed successfully!')

# Clean up working directory
shutil.rmtree(working_directory)
print("\nCleaned up working directory")

# Record end time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")