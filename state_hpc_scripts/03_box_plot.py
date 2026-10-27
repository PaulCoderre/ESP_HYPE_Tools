import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import networkx as nx
from properscoring import crps_ensemble

# Inputs
geodata= pd.read_csv('../state_model/GeoData.txt', index_col=0, sep='\t')

fca = pd.read_csv('../hist_data/timeFCON_DD.txt', sep= '\t', skiprows=1, index_col=0)

cilv = pd.read_csv('../hist_data/timeCILV_DD.txt', sep= '\t', skiprows=1, index_col=0)

prec= pd.read_csv('../hist_data/timeCPRC_DD.txt', sep= '\t',skiprows=1, index_col=0)

plot_configs = [
    {
        'dir_path': './esp_init_results/apr1_start_sep30_end/58208_esp_results/',
        'computed_path': '../hist_data/full/0058208.txt',
        'start_date': '04-01',
        'end_date': '07-31',
        'title': 'St. Mary: SMRIB',
        'gauge_id': 58208,
    },
        {
        'dir_path': './esp_init_results/apr1_start_sep30_end/58213_esp_results/',
        'computed_path': '../hist_data/full/0058213.txt',
        'start_date': '04-01',
        'end_date': '07-31',
        'title': 'St. Mary: SMRBB',
        'gauge_id': 58213,
    },
    {
        'dir_path': './esp_init_results/feb1_start_sep30_end/58290_esp_results/',
        'computed_path': '../hist_data/full/0058290.txt',
        'start_date': '02-01',
        'end_date': '05-31',
        'title': 'Milk River: FRRIB',
        'gauge_id': 58290,
    },
        {
        'dir_path': './esp_init_results/feb1_start_sep30_end/58308_esp_results/',
        'computed_path': '../hist_data/full/0058308.txt',
        'start_date': '02-01',
        'end_date': '05-31',
        'title': 'Milk River: MREIB',
        'gauge_id': 58308,
    },
        {
        'dir_path': './esp_init_results/feb1_start_sep30_end/58356_esp_results/',
        'computed_path': '../hist_data/full/0058356.txt',
        'start_date': '02-01',
        'end_date': '05-31',
        'title': 'Milk River: LDCIB',
        'gauge_id': 58356,
    },
        {
        'dir_path': './esp_init_results/feb1_start_sep30_end/58363_esp_results/',
        'computed_path': '../hist_data/full/0058363.txt',
        'start_date': '02-01',
        'end_date': '05-31',
        'title': 'Milk River: BTCIB',
        'gauge_id': 58363,
    },


]

output_path = './new_plots/'

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import os
import pandas as pd
import numpy as np
import xarray as xr

SECONDS_PER_DAY = 86400

# Climatology period -- same convention as the semi-monthly CRPSS script
# (fig5_crpss_by_leadtime), so the two CRPSS numbers are directly
# comparable in scope even though this one is computed on the FULL
# 4-month freshet window instead of semi-monthly windows.
CLIM_START_YEAR = 1985
CLIM_END_YEAR   = 2015


def compute_crpss_for_window(sim, sum_cout_series_combined, sum_model_sim_combined,
                              years_sorted, start_month, start_day, end_month, end_day):
    """
    CRPSS for the FIXED full-window (e.g. Apr 1-Jul 31) volume, using the
    SAME construction as fig5_crpss_by_leadtime:
      - CRPS_esp: ESP ensemble members for that hindcast year vs. the
        "perfect"/simulated value for that year.
      - CRPS_clim: a leave-one-out climatology ensemble built by summing
        the full historical simulated series over the SAME calendar
        window in every other year (1985-2015, excluding the forecast
        year), with a leap-year-safe end-day clip identical to the
        semi-monthly script.
      - CRPSS = 1 - mean(CRPS_esp) / mean(CRPS_clim), with the mean taken
        across hindcast years BEFORE forming the ratio (matches how the
        semi-monthly script aggregates per-period CRPS across years
        before computing CRPSS from the ratio of means, not a mean of
        per-year CRPSS values).
    Units: CRPS_esp/CRPS_clim are both computed on volume (cout *
    SECONDS_PER_DAY, summed over the window) -- the climatology values
    below apply the same SECONDS_PER_DAY conversion as the ESP/sim series
    already carry, so the two sides of the ratio stay on the same scale.
    """
    crps_esp_list, crps_clim_list = [], []

    for year in years_sorted:
        ensemble_vals = sum_cout_series_combined[sum_cout_series_combined.index.year == year].values
        sim_vals_this_year = sum_model_sim_combined[sum_model_sim_combined.index.year == year].values
        if len(ensemble_vals) == 0 or len(sim_vals_this_year) == 0:
            continue
        sim_val = sim_vals_this_year[0]  # constant across members for a given year

        crps_esp = crps_ensemble(sim_val, ensemble_vals)

        all_years = pd.Series(sim.index.year.unique())
        climatology_years = all_years[
            (all_years >= CLIM_START_YEAR) &
            (all_years <= CLIM_END_YEAR) &
            (all_years != year)
        ].values

        clim_ensemble = []
        for y in climatology_years:
            try:
                y_start = pd.Timestamp(year=y, month=start_month, day=start_day)
                # Leap-year-safe end-day clip, same as fig5_crpss_by_leadtime.
                y_end_day = min(end_day, pd.Timestamp(year=y, month=end_month, day=1).days_in_month)
                y_end = pd.Timestamp(year=y, month=end_month, day=y_end_day)
                clim_val = (sim.loc[y_start:y_end]['cout'] * SECONDS_PER_DAY).sum()
                clim_ensemble.append(clim_val)
            except Exception:
                continue

        if len(clim_ensemble) < 3:
            continue

        crps_clim = crps_ensemble(sim_val, clim_ensemble)

        crps_esp_list.append(crps_esp)
        crps_clim_list.append(crps_clim)

    if not crps_esp_list:
        return np.nan

    mean_crps_esp = np.mean(crps_esp_list)
    mean_crps_clim = np.mean(crps_clim_list)
    if mean_crps_clim == 0:
        return np.nan
    return 1 - (mean_crps_esp / mean_crps_clim)


def process_and_plot(config, ax):
    directory_path = config['dir_path']
    computed_path = config['computed_path']
    start_date = config['start_date']
    end_date = config['end_date']
    plot_title = config['title']

    start_month, start_day = map(int, start_date.split('-'))
    end_month, end_day = map(int, end_date.split('-'))

    all_sum_cout_series = []
    sum_model_sim = []

    # Read observed full-model simulation
    sim = pd.read_csv(computed_path, sep='\t', index_col=0)
    sim = sim.drop('UNITS', axis=0)
    sim.index = pd.to_datetime(sim.index, errors='coerce')
    sim['cout'] = pd.to_numeric(sim['cout'], errors='coerce')

    for filename in os.listdir(directory_path):
        if filename.endswith('.nc'):
            file_path = os.path.join(directory_path, filename)
            esp = xr.open_dataset(file_path).astype(float)
            esp['DATE'] = pd.to_datetime(esp['DATE'])

            last_year = pd.to_datetime(esp['DATE'][-1].values).year
            start_dt = pd.Timestamp(year=last_year, month=start_month, day=start_day)
            end_dt = pd.Timestamp(year=last_year, month=end_month, day=end_day)

            ds_selected = esp.sel(DATE=slice(start_dt, end_dt))
            sum_cout = (ds_selected['cout'] * SECONDS_PER_DAY).sum(dim='DATE')
            sum_cout_series = sum_cout.to_series()
            sum_cout_series.index = pd.to_datetime(sum_cout_series.index)
            sum_cout_series.index = sum_cout_series.index.map(lambda x: x.replace(year=last_year))
            all_sum_cout_series.append(sum_cout_series)

            model_sim = sim.loc[start_dt:end_dt]
            sum_sim = (model_sim['cout'] * SECONDS_PER_DAY).sum()
            sim_series = sum_cout_series.copy()
            sim_series[:] = sum_sim
            sim_series.name = 'sim'
            sum_model_sim.append(sim_series)

    if not all_sum_cout_series:
        ax.set_title(f"{plot_title}\n(No .nc files found)", fontsize=10)
        return []

    sum_cout_series_combined = pd.concat(all_sum_cout_series)
    sum_model_sim_combined = pd.concat(sum_model_sim)

    years_sorted = sorted(sum_cout_series_combined.index.year.unique())
    data = [sum_cout_series_combined[sum_cout_series_combined.index.year == year].values for year in years_sorted]

    ax.boxplot(data, patch_artist=True, medianprops=dict(color='red'),
               boxprops=dict(color='black', facecolor='lightgray'))

    ax.set_xticks(range(1, len(years_sorted)+1))
    ax.set_xticklabels(years_sorted, rotation=45, fontsize=9)
    ax.set_title(plot_title, fontsize=11)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(f'Total Streamflow Volume\n({start_month:02d}-{start_day:02d} to {end_month:02d}-{end_day:02d}) [m³]', fontsize=10)

    # --- CRPSS for the fixed full-window volume, same logic as fig5_crpss_by_leadtime ---
    crpss_val = compute_crpss_for_window(sim, sum_cout_series_combined, sum_model_sim_combined,
                                          years_sorted, start_month, start_day, end_month, end_day)
    if not np.isnan(crpss_val):
        ax.text(0.97, 0.95, f'CRPSS = {crpss_val:.2f}', transform=ax.transAxes,
                fontsize=9, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=3))

    outlier_records = []
    for i, year in enumerate(years_sorted, start=1):
        year_data = sum_cout_series_combined[sum_cout_series_combined.index.year == year].values
        q1, q3 = np.percentile(year_data, 25), np.percentile(year_data, 75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    
        y_sim = sum_model_sim_combined[sum_model_sim_combined.index.year == year].values
        x_sim = np.random.normal(i, 0.00001, size=len(y_sim))
        inliers = (y_sim >= lower_bound) & (y_sim <= upper_bound)
    
        ax.plot(x_sim[inliers], y_sim[inliers], 'b.', alpha=0.5, markersize=10)
        ax.plot(x_sim[~inliers], y_sim[~inliers], 'r.', alpha=0.5, markersize=10)
    
        median_val = np.median(year_data)
        if np.any(~inliers):
            val = y_sim[0]
            outlier_records.append({
                'title': plot_title,
                'year': year,
                'sim_value': val,
                'median': median_val,
                'difference': val - median_val,
            })

    return outlier_records


# -----------------------------
# Generate Subplots -- 2-col grid
# -----------------------------
ncols = 2
nrows = int(np.ceil(len(plot_configs) / ncols))

fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols,
    figsize=(16, 4.5 * nrows),
    constrained_layout=True
)

axes_flat = axes.flatten()

all_outlier_info = []
for i, (ax, config) in enumerate(zip(axes_flat, plot_configs)):
    print(f'Processing {i} - {config["title"]}')
    outliers = process_and_plot(config, ax)
    all_outlier_info.extend(outliers)
    ax.set_xlabel('Year', fontsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

for ax in axes_flat[len(plot_configs):]:
    ax.set_visible(False)

# No subplots_adjust needed -- constrained_layout handles it

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Simulated (Inlier)',
           markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Simulated (Outlier)',
           markerfacecolor='red', markersize=8)
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))

os.makedirs(output_path, exist_ok=True)
fig.savefig(os.path.join(output_path, 'fig4_esp_boxplot.png'), dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(output_path, 'fig4_esp_boxplot.pdf'), format='pdf', bbox_inches='tight')
plt.show()