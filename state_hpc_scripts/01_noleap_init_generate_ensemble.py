#!/usr/bin/env python
# coding: utf-8

# ## Description
# ____________
# This script generates ESP forcing files for the forecast period only (no spin-up).
# Assumes model states will be loaded from a pre-saved state file at the forecast initialization date.

# ### Import Libraries

import os
import pandas as pd
from datetime import datetime
import xarray as xr
import numpy as np
import calendar
import time
from calendar import isleap

# ### Inputs
# ====================
# esp ensemble decisions
start_date = '04-01'                        # start month and day for esp forecast in %m-%d
end_date = '09-30'                          # end month and day for esp forecast in %m-%d

forcing_files = ['Tobs.txt', 'Pobs.txt', 'TMAXobs.txt', 'TMINobs.txt']  # HYPE forcing files
hype_directory = '../state_model'                  # location of HYPE model files
output_directory = './esp_init_forcings/apr1_start_sep30_end_no_spinup/'  # output directory

# Alternative Input - Create forcings for every year in a range
start_year = 1985
end_year = 2015
esp_year = [str(year) for year in range(start_year, end_year + 1)]
print(f"Generating ESP forcings for years: {esp_year}")

# NEW: Specify year range for ensemble members
ensemble_start_year = 1985  # First year to use for ensemble members
ensemble_end_year = 2015    # Last year to use for ensemble members
print(f"Ensemble members will be drawn from years: {ensemble_start_year}-{ensemble_end_year}")

# =====================
# ### Generate Ensemble

# Record start time
start_time = time.time()

# Save the current working directory path
current_directory = os.getcwd()

# Extract month and day from the dates
start_month, start_day = map(int, start_date.split('-'))
end_month, end_day = map(int, end_date.split('-'))

# Create the parent directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)
print(f"Output directory: '{output_directory}'")

# Iterate through each analysis year
for analysis_year in esp_year:
    
    # Define the output directory name for the current year
    year_directory = os.path.join(output_directory, f"{analysis_year}_forcings")
    
    # Create the year directory if it doesn't exist
    os.makedirs(year_directory, exist_ok=True)
    print(f"\nProcessing year {analysis_year}...")
    
    # Convert year to int
    analysis_year_int = int(analysis_year)
    
    for file_name in forcing_files:
        
        # Read forcing file
        forcing = pd.read_csv(os.path.join(hype_directory, file_name), 
                             index_col=0, sep='\t')
        
        # Set the time series in the index to datetime
        forcing.index = pd.to_datetime(forcing.index)
        
        # Extract unique years from the datetime index (excluding analysis year)
        unique_years = forcing.index.year.unique()
        unique_years = unique_years[unique_years != analysis_year_int]

        # NEW: Filter to only include years within the specified ensemble range
        unique_years = unique_years[
            (unique_years >= ensemble_start_year) & 
            (unique_years <= ensemble_end_year) & 
            (unique_years != analysis_year_int)
        ]
        
        print(f"  {file_name}: Creating ensemble from {len(unique_years)} historical years")
        
        # Create an empty list to store ensemble_member DataArrays
        ensemble_member_list = []
        
        # Create the ensemble by iterating through each historical year
        for i, year in enumerate(unique_years):
            
            # Extract forecast period for current year
            start_date_year = pd.Timestamp(year, start_month, start_day)
            end_date_year = pd.Timestamp(year, end_month, end_day)
            
            year_data = forcing.loc[start_date_year:end_date_year].copy()
            
            # Remove Feb 29 if it exists
            feb_29_mask = year_data.index.strftime('%m-%d') == '02-29'
            if feb_29_mask.any():
                year_data = year_data.loc[~feb_29_mask]
                print(f"    Removed Feb 29 from year {year}")
            
            # Change the year to the analysis year
            year_data.index = year_data.index.map(lambda x: x.replace(year=analysis_year_int))
            
            # Handle leap year if analysis year is a leap year
            expected_dates = pd.date_range(
                start=pd.Timestamp(analysis_year_int, start_month, start_day),
                end=pd.Timestamp(analysis_year_int, end_month, end_day),
                freq='D'
            )
            
            # Check for missing dates (Feb 29 in leap years)
            missing_dates = expected_dates.difference(year_data.index)
            
            if not missing_dates.empty:
                # Reindex and backfill missing dates
                year_data = year_data.reindex(expected_dates).bfill()
                print(f"    Backfilled {len(missing_dates)} missing date(s) for ensemble member {i}")
            
            # Create an xarray DataArray from the ensemble member
            data_array = xr.DataArray(
                year_data.values,
                dims=('time', 'subbasin'),
                coords={
                    'time': year_data.index,
                    'subbasin': year_data.columns,
                    'ensemble_member': i
                }
            )
            
            # Append the DataArray to the list
            ensemble_member_list.append(data_array)
        
        # Concatenate ensemble_member DataArrays along the ensemble_member dimension
        final_data_array = xr.concat(ensemble_member_list, dim='ensemble_member')
        
        # Save the DataArray to a netCDF file for the current year
        output_path = os.path.join(year_directory, f"{file_name.split('.')[0]}.nc")
        final_data_array.to_netcdf(output_path)
        print(f"    Saved: {output_path}")
    
    print(f"  {analysis_year} forcings complete!")

# Record end time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")