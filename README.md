# ESP_HYPE_Tools
This repository contains scripts to perform Ensemble Streamflow Prediction for the HYPE model and visualize and analyze the results.  

## Contents
### Model  
This is an empty directory where all requred files to run the HYPE model can be copied. 

### Scripts
This directory contains Jupyter Notebooks capable of creating a forcing ensemble for ESP, running the ensemble, visualizing the outputs in different ways and calculating statistics. 

### swe_analysis
This directory contains scripts for establishing a correlation between snow water equivalent (SWE) and summer runoff in snowmelt dominated basins. This may be useful in establishing the effectiveness of ESP in these basins. 

### sample_outputs
This directory contains sample plots of ESP analysis and tables of statistics that can be generated.

## Instructions
1. Copy HYPE model into model directory.
2. Run 01_generate_ensemble.ipynb to create the ensemble of forcings. The ensemble is created from every year in the model forcing files (not including the year of analysis). Required inputs are:
   - Year of analysis
   - Start date for model spin up. Default is the start of the water year (October 1st)
   - Start and end date of ESP period. Default is April 1st to September 29th.
   - List of the forcing files in the HYPE model. Default is Pobs.txt, Tobs.txt, TMAXobs.txt and TMINobs.txt.
   - Directory containing the model. Default is ../model.
   - Output directory. Default is ../esp_forcings.
3. Run 02_run_ensemble.ipynb to run the model for each year of analysis and each ensemble member within it. It saves the outputs as .nc files. Requirements:
   - Set resultsdir in info.txt to ./ (current directory)
   - Define a basin output file for the subbasin to be analyzed containing variables cout, rout and snow representing simulated flow, observed flow and snow water equivalent
      Required inputs are:
   - Subbasin ID to be analyzed.
   - Command for running HYPE.
   - Directory containing the ESP forcings. Default is ../esp_forcings
   - Directory containing the model. Default is ../model.
   - Output directory. Default is ./
4. Run 03_visualize_outputs.ipynb to create a plot for a given year containing each ensemble member, uncertainty bands and the mean of the ensemble for a given variable. Required inputs are:
   - Output file to plot
   - variable to analyze. Can be cout or snow representing simulated streamflow and SWE
   - Output directory. Default is ./
   - (**Optionally**) Path to a full HYPE simulation including the analysis year. If specified, the script will add computed runoff for known historical forcings to the plot. Useful for validating the ESP performance against simulated runoff.
5. Run 04_bar&whisker_plot.ipynb to generate a bar and whisker plot comparing the ensemble to observed streamflow for each year included in the analysis. Useful for comparing the ESP performance between different years. Required inputs are:
   - Path to the directory containing the ESP outputs.
   - Start and end date for the ESP analysis. These are the dates between which the quartiles will be calculated for the ensemble. Changing this allows for performance comparison between different lead times.
   - Output directory. Default is ./
   - (**Optionally**) Path to a full HYPE simulation including the analysis year. If specified, the script will add computed runoff for known historical forcings to the plot.
6. Run 05_statistics.ipynb script to output tables containing the bias for each year included in the ESP analysis, correlation coefficient, RMSE and NSE (Huang et al. 2017). These statistics are being calculated with the mean of the ensemble. More statistics can be added as needed. Required inputs are:
   - Path to the directory containing the ESP outputs.
   - Start and end date for the ESP analysis. These are the dates between which the quartiles will be calculated for the ensemble. Changing this allows for performance comparison between different lead times.
   - Output directory. Default is ./
   - (**Optionally**) Path to a full HYPE simulation including the analysis year. If specified, the script will add computed runoff for known historical forcings to the plot.

   

