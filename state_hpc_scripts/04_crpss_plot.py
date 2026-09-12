#!/usr/bin/env python
# coding: utf-8
# ## Description
# This script calculates CRPS and CRPSS for ESP forecasts at different lead times

# ### Import Libraries
import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from properscoring import crps_ensemble
from pandas.tseries.offsets import DateOffset

# ### Inputs
forecast_runs = [
    ('02-01', './esp_init_results/feb1_start_sep30_end/'),
    ('03-01', './esp_init_results/mar1_start_sep30_end/'),
    ('04-01', './esp_init_results/apr1_start_sep30_end/')
]

# Define inputs for the plot
subbasin_ids = ['58208', '58213', '58290', '58308', '58356', '58363']
variable = 'cout'
output_directory = './new_plots/'

# Climatology period (inclusive)
clim_start_year = 1985
clim_end_year   = 2015

# ============================================
# LEAD TIMES TO CALCULATE AND PLOT
# ============================================
# lead_times = {
#     '4w': DateOffset(weeks=4),   # ~1 month
#     '8w': DateOffset(weeks=8),   # ~2 months
#     '12w': DateOffset(weeks=12), # ~3 months
#     '16w': DateOffset(weeks=16), # ~4 months
#     '20w': DateOffset(weeks=20), # ~5 months
#     '24w': DateOffset(weeks=24)  # ~6 months
# }
# lead_order = list(lead_times.keys())
lead_times = {f'{w}w': DateOffset(weeks=w) for w in range(1, 27)}
lead_order = list(lead_times.keys())

# Path to observed/simulated data
computed_path = '../hist_data/full/'


# ### Compute CRPS
def compute_crpss_for_start(start_date_str, directory_path, computed_path, subbasin_ids, variable,
                             clim_start_year, clim_end_year):
    start_month, start_day = map(int, start_date_str.split('-'))
    crps_global_list = []

    for subbasin_id in subbasin_ids:
        print(f"\nProcessing subbasin {subbasin_id}...")

        # --- 1. Load Observed Data ---
        matching_txt_files = [f for f in os.listdir(computed_path) if f.endswith('.txt') and subbasin_id in f]
        if not matching_txt_files:
            print(f"  No .txt file found for subbasin {subbasin_id}, skipping.")
            continue
        sim_file_path = os.path.join(computed_path, matching_txt_files[0])
        sim = pd.read_csv(sim_file_path, sep='\t', index_col=0, low_memory=False).apply(pd.to_numeric, errors='coerce')
        if 'UNITS' in sim.index:
            sim = sim.drop('UNITS', axis=0)
        sim.index = pd.to_datetime(sim.index, errors='coerce')
        sim = sim[[variable]]

        # --- 2. Load Forecast Ensemble Files ---
        matching_subdirs = [d for d in os.listdir(directory_path)
                            if os.path.isdir(os.path.join(directory_path, d)) and subbasin_id in d]
        if not matching_subdirs:
            print(f"  No subdirectory found for subbasin {subbasin_id}, skipping.")
            continue
        subdir_path = os.path.join(directory_path, matching_subdirs[0])
        files_for_subbasin = [f for f in sorted(os.listdir(subdir_path)) if f.endswith('.nc')]
        if not files_for_subbasin:
            print(f"  No .nc files found in {subdir_path}, skipping.")
            continue

        for filename in files_for_subbasin:
            file_path = os.path.join(subdir_path, filename)
            esp = xr.open_dataset(file_path)
            esp['DATE'] = pd.to_datetime(esp['DATE'].values)
            esp = esp.astype(float)
            first_date = pd.to_datetime(esp['DATE'][0].values)
            base_year = first_date.year

            base_start = pd.Timestamp(year=base_year, month=start_month, day=start_day)

            for label, offset in lead_times.items():
                start_date = base_start
                end_date = start_date + offset - pd.Timedelta(days=1)
                if end_date > esp['DATE'].values[-1]:
                    continue

                # --- ESP ensemble sum ---
                ds_selected = esp.sel(DATE=slice(start_date, end_date))
                sum_var_series = ds_selected[variable].sum(dim='DATE').to_series()

                # --- Observed sum ---
                sum_obs = sim.loc[start_date:end_date][variable].sum()

                # --- Climatology ensemble (1985-2015, excluding base_year) ---
                all_years = pd.Series(sim.index.year.unique())
                climatology_years = all_years[
                    (all_years >= clim_start_year) &
                    (all_years <= clim_end_year) &
                    (all_years != base_year)
                ].values

                clim_ensemble = []
                for y in climatology_years:
                    try:
                        y_start = pd.Timestamp(year=y, month=start_date.month, day=start_date.day)
                        y_end   = pd.Timestamp(year=y, month=end_date.month,   day=end_date.day)
                        clim_val = sim.loc[y_start:y_end][variable].sum()
                        clim_ensemble.append(clim_val)
                    except:
                        continue

                if len(clim_ensemble) < 3:
                    print(f"  Not enough climatology members for {subbasin_id}, {base_year}, {label}")
                    continue

                # --- Compute CRPS ---
                crps_esp  = crps_ensemble(sum_obs, sum_var_series.values)
                crps_clim = crps_ensemble(sum_obs, clim_ensemble)
                crpss_val = 1 - (crps_esp / crps_clim)

                print(f"  Subbasin: {subbasin_id} | Year: {base_year} | Lead: {label:3s} | "
                      f"N_clim: {len(clim_ensemble):2d} | "
                      f"CRPS_ESP: {crps_esp:8.3f} | CRPS_CLIM: {crps_clim:8.3f} | CRPSS: {crpss_val:+.3f}")

                crps_global_list.append({
                    'subbasin_id': subbasin_id,
                    'year': base_year,
                    'lead_time': label,
                    'crps_esp': crps_esp,
                    'crps_clim': crps_clim,
                })

    df = pd.DataFrame(crps_global_list)
    df['forecast_start'] = pd.to_datetime(f"{base_year}-{start_date_str}").strftime('%b %d')
    return df


# Calculate CRPS for all forecast runs
all_crps_df = pd.DataFrame()
for start_date, path in forecast_runs:
    df = compute_crpss_for_start(start_date, path, computed_path, subbasin_ids, variable,
                                  clim_start_year, clim_end_year)
    all_crps_df = pd.concat([all_crps_df, df], ignore_index=True)

all_crps_df.to_csv(os.path.join(output_directory, 'crps_dataframe.csv'))

# -----------------------------
# Color & style setup
# -----------------------------
subbasin_colors = {
    '58208': '#E69F00',  # SMRIB - orange
    '58213': '#56B4E9',  # SMRBB - sky blue
    '58223': '#009E73',  # SWCSB - green
    '58408': '#F0E442',  # MRWIB - yellow
    '58308': '#0072B2',  # MREIB - blue
    '58290': '#D55E00',  # FRRIB - red-orange
    '58356': '#CC79A7',  # LDCIB - pink/magenta
    '58363': '#999999',  # BTCIB - gray
    '58242': 'black'     # Outlet
}

hype_id_to_location = {
    '58213': ('St. Mary', 'SMRBB'),
    '58208': ('St. Mary', 'SMRIB'),
    '58223': ('St. Mary', 'SWCSB'),
    '58308': ('Milk', 'MREIB'),
    '58408': ('Milk', 'MRWIB'),
    '58363': ('Milk', 'BTCIB'),
    '58290': ('Milk', 'FRRIB'),
    '58356': ('Milk', 'LDCIB'),
    '58242': ('Milk', 'Outlet')
}


# ============================================
# HELPER — monthly tick positions per init date
# ============================================
def get_monthly_ticks(init_month, init_day, n_weeks=26):
    """
    Returns (positions, labels) for the 1st of each month that falls
    within [1 week, n_weeks weeks] of the initialisation date.
    Positions are fractional indices into a 0-based weekly array.
    """
    init_date = pd.Timestamp(year=2001, month=init_month, day=init_day)
    ticks, labels = [], []
    for offset in range(1, 9):          # look up to 8 months ahead
        target = (init_date + DateOffset(months=offset)).replace(day=1)
        days   = (target - init_date).days
        pos    = days / 7.0 - 1.0       # index 0 = week 1 = 7 days out
        if -0.5 <= pos <= n_weeks - 0.5:
            ticks.append(pos)
            labels.append(target.strftime('%b 1'))
    return ticks, labels

# Define subbasins
all_subbasins = ['58208', '58213', '58290', '58308', '58356', '58363']
    
# Define which subbasins belong to which river system
st_mary_subbasins     = ['58208', '58213']
milk_river_subbasins  = ['58290', '58308', '58356', '58363']

# ============================================
# PLOTTING
# ============================================
n_weeks = 26
init_info = [
    ('Feb 01', 2, 1),
    ('Mar 01', 3, 1),
    ('Apr 01', 4, 1),
]

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)   # ← sharex=False

for idx, (ax, (forecast_start, init_m, init_d)) in enumerate(zip(axes, init_info)):
    for subbasin_id in all_subbasins:
        crpss_values = []
        for lead_week in lead_order:
            lead_data = all_crps_df[
                (all_crps_df['subbasin_id'] == subbasin_id) &
                (all_crps_df['forecast_start'] == forecast_start) &
                (all_crps_df['lead_time'] == lead_week)
            ].copy()
            if len(lead_data) > 0:
                mean_crps_esp  = lead_data['crps_esp'].mean()
                mean_crps_clim = lead_data['crps_clim'].mean()
                crpss = 1 - (mean_crps_esp / mean_crps_clim)
                crpss_values.append(crpss)
            else:
                crpss_values.append(np.nan)

        if subbasin_id in st_mary_subbasins:
            linestyle, linewidth = '-', 2.0
        else:
            linestyle, linewidth = '--', 2.5

        basin, basin_label = hype_id_to_location.get(subbasin_id, ('Unknown', subbasin_id))

        if subbasin_id == '58408':
            ax.plot(range(n_weeks), crpss_values,
                    color=subbasin_colors[subbasin_id],
                    linestyle=linestyle, linewidth=linewidth + 0.5,
                    marker='o', markersize=7,
                    markeredgecolor='black', markeredgewidth=0.8,
                    label=f"{basin}: {basin_label}", alpha=0.9)
        else:
            ax.plot(range(n_weeks), crpss_values,
                    color=subbasin_colors[subbasin_id],
                    linestyle=linestyle, linewidth=linewidth,
                    marker='o', markersize=6,
                    label=f"{basin}: {basin_label}", alpha=0.8)

    ax.axhline(0, color='black', linestyle='-', linewidth=1.5)    # ← solid, no alpha

    ax.set_ylabel('CRPSS', fontsize=12, fontweight='bold')
    ax.set_title(f'{forecast_start} Forecast Initialization', fontsize=12, fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.5, n_weeks - 0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Dynamic monthly x-axis ticks ---
    ticks, labels = get_monthly_ticks(init_m, init_d, n_weeks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=11)

    if idx == 0:
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2)
    if idx == 2:
        ax.set_xlabel('Forecast Date', fontsize=12, fontweight='bold')

fig.suptitle('CRPSS by Lead Time and Initialization Date\n(Solid = St. Mary, Dashed = Milk River)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
os.makedirs(output_directory, exist_ok=True)
plt.savefig(os.path.join(output_directory, 'fig5_crpss_by_leadtime.png'), dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(output_directory, 'fig5_crpss_by_leadtime.pdf'), format='pdf', bbox_inches='tight')
plt.show()
