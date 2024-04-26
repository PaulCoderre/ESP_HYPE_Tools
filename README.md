# ESP_HYPE_Tools
This repository contains scripts to perform Ensemble Streamflow Prediction (ESP) for the HYPE model and visualize and analyze the results.  

## Contents
### Model  
This is an empty directory where all requred files to run the HYPE model can be copied. 

### Scripts
This directory contains Jupyter Notebooks capable of creating a forcing ensemble for ESP, running the ensemble, visualizing the outputs in different ways and calculating statistics. 

### swe_analysis
This directory contains scripts for establishing a correlation between snow water equivalent (SWE) and summer runoff in snowmelt dominated basins. This may be useful in establishing the effectiveness of ESP in these basins. 

### sample_outputs
This directory contains sample plots of ESP analysis and tables of statistics that can be generated.

## ESP Instructions
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
   - (**Optional**) Path to a full HYPE simulation including the analysis year. If specified, the script will add computed runoff for known historical forcings to the plot. Useful for validating the ESP performance against simulated runoff.
5. Run 04_bar&whisker_plot.ipynb to generate a bar and whisker plot comparing the ensemble to observed streamflow for each year included in the analysis. Useful for comparing the ESP performance between different years. Required inputs are:
   - Path to the directory containing the ESP outputs.
   - Start and end date for the ESP analysis. These are the dates between which the quartiles will be calculated for the ensemble. Changing this allows for performance comparison between different lead times.
   - Output directory. Default is ./
   - (**Optional**) Path to a full HYPE simulation including the analysis year. If specified, the script will add computed runoff for known historical forcings to the plot.
6. Run 05_statistics.ipynb script to output tables containing the bias for each year included in the ESP analysis, correlation coefficient, RMSE and NSE (Huang et al. 2017). These statistics are being calculated with the mean of the ensemble. More statistics can be added as needed. Required inputs are:
   - Path to the directory containing the ESP outputs.
   - Start and end date for the ESP analysis. These are the dates between which the quartiles will be calculated for the ensemble. Changing this allows for performance comparison between different lead times.
   - Output directory. Default is ./
   - (**Optional**) Path to a full HYPE simulation including the analysis year. If specified, the script will add computed runoff for known historical forcings to the plot.

## SWE Analysis Instructions
1. Download desired SWE data. This workflow was designed for a 4 km gridded SWE and snow depth dataset (Broxton et al., 2019) that was created by assimilating PRISM daily temperature and precipitation data, SWE and snow depth data from the SNOTEL station network and snow depth data from the COOP network.
2. Run process_snotel.ipynb script to reproject, trim extents and add a calendar type. Modify as required.
3. Run the EASYMORE workflow (Gharari et al., 2023) to remap the SWE data to a given shapefile. A sample EASYMORE script is provided.
4. Run swe&runoff.ipynb script to rank each year's SWE and summer runoff and plot the results. The r^2 is calculated for the plot. Required inputs are:
   - Path to the EASYMORE outputs.
   - Start and end date for the Analysis.
   - Path to HYPE's observed flow input file (Qobs.txt).
   - River segment to be analyzed.
   - Location of output directory. 

## References
Broxton, P., X. Zeng, and N. Dawson. (2019). Daily 4 km Gridded SWE and Snow Depth from Assimilated In-Situ and Modeled Data over the Conterminous US, Version 1 [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/0GGPB220EX6A. Date Accessed 04-26-2024.

Gharari, S., Keshavarz, K., Knoben, W. J. M., Tang, G., & Clark, M. P. (2023). EASYMORE: A Python package to streamline the remapping of variables for Earth System models. SoftwareX, 24, 101547. https://doi.org/10.1016/j.softx.2023.101547

Huang, C., Newman, A. J., Clark, M. P., Wood, A. W., & Zheng, X. (2017). 1 Evaluation of snow data assimilation using the Ensemble Kalman 2 Filter for seasonal streamflow prediction in the Western United 3 States. Hydrology and Earth System Sciences, 21(1), 635–650. https://hess.copernicus.org/preprints/hess-2016-185/hess-2016-185-manuscript-version6.pdf


   

