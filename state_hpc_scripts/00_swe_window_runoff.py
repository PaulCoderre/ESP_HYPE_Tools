#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import math
import matplotlib as mpl
import numpy as np
import os
from pandas.tseries.offsets import DateOffset
import matplotlib.gridspec as gridspec
from datetime import datetime
import warnings
from matplotlib.cm import get_cmap


# In[2]:


# Inputs
geodata = pd.read_csv('../state_model/GeoData.txt', index_col=0, sep='\t')
swe  = pd.read_csv('../hist_data/timeUPSNOW_DD.txt', sep='\t', skiprows=1, index_col=0)
cilv = pd.read_csv('../hist_data/timeCILV_DD.txt', sep='\t', skiprows=1, index_col=0)
prec = pd.read_csv('../hist_data/timeCPRC_DD.txt', sep='\t', skiprows=1, index_col=0)
cout = pd.read_csv('~/research/SMM_Models/hype/model/v10_model/final_model/hds_v3/results/timeCOUT_DD.txt', sep='\t', skiprows=1, index_col=0)
output_path = "./window_plots/"

# Analysis period
date_start = "1985-01-01"
date_end   = "2015-12-31"

# Per-region streamflow windows for the SCATTER plots -- back to the FULL
# 4-month snowmelt-season window per region, separate from the semi-monthly
# breakdown in the bottom panel (same restructuring as the pothole-storage
# script): top panel answers "how well does antecedent SWE predict the
# FULL freshet volume," bottom panel answers "how does that skill vary
# across individual 2-week periods."
season_windows = {
    'Milk River': {'start': (2, 1), 'end': (5, 31)},
    'St. Mary':   {'start': (4, 1), 'end': (7, 31)},
}

# Gauges
gauge_ids = [58208, 58213, 58290, 58308, 58356, 58363]

hype_id_to_location = {
    '0058328': ('Milk River', 'BCBMO'),
    '0058346': ('Milk River', 'BSCMO'),
    '0058363': ('Milk River', 'BTCIB'),
    '0058425': ('Milk River', 'CLCMO'),
    '0058290': ('Milk River', 'FRRIB'),
    '0058356': ('Milk River', 'LDCIB'),
    '0058408': ('Milk River', 'MRWIB'),
    '0058643': ('Milk River', 'NFKMR'),
    '0058418': ('Milk River', 'PPCMO'),
    '0058398': ('Milk River', 'RKCMO'),
    '0058213': ('St. Mary', 'SMRBB'),
    '0058208': ('St. Mary', 'SMRIB'),
    '0058223': ('St. Mary', 'SWCSB'),
    '0058308': ('Milk River', 'MREIB'),
    '0058242': ('Milk River', 'Outlet')
}


# In[3]:


# Format data
geodata['maindown'] = geodata['maindown'].astype(int)
geodata.index = geodata.index.astype(int)

swe.index  = pd.to_datetime(swe.index)
cilv.index = pd.to_datetime(cilv.index)
prec.index = pd.to_datetime(prec.index)
cout.index = pd.to_datetime(cout.index)

swe.columns  = swe.columns.astype(int)
cilv.columns = cilv.columns.astype(int)
prec.columns = prec.columns.astype(int)
cout.columns = cout.columns.astype(int)

subbasin_area = geodata["area"].to_dict()

swe  = swe.loc[date_start:date_end]
cilv = cilv.loc[date_start:date_end]
prec = prec.loc[date_start:date_end]
cout = cout.loc[date_start:date_end]


# ### Scatter Plot

# In[4]:


# Helper: look up region for a gauge
def get_gauge_region(gauge_id):
    gauge_str = str(gauge_id).zfill(7)
    return hype_id_to_location[gauge_str][0] if gauge_str in hype_id_to_location else None

# Antecedent date per region:
#   Milk River → Feb 1 (pre-melt snowpack)
#   St. Mary   → Apr 1 (later melt timing)
start_year = pd.to_datetime(date_start).year
end_year   = pd.to_datetime(date_end).year

antecedent_dates = {
    'Milk River': pd.to_datetime([f"{y}-02-01" for y in range(start_year, end_year + 1)]),
    'St. Mary':   pd.to_datetime([f"{y}-04-01" for y in range(start_year, end_year + 1)]),
}
antecedent_month_day = {
    'Milk River': (2, 1),
    'St. Mary':   (4, 1),
}
# Antecedent date label per region (for axis labels)
antecedent_date_label = {
    'Milk River': 'Feb 1',
    'St. Mary':   'Apr 1',
}


# ============================================
# SEMI-MONTHLY (15-DAY) SLIDING WINDOWS
# One universal calendar-aligned period grid (Feb 1-15, Feb 16-28/29, ...,
# Sep 16-30), shared by both regions since both antecedent dates (Feb 1,
# Apr 1) land on a period boundary. Milk River is evaluated from period 0;
# St. Mary only from the period starting Apr 1 onward (earlier periods left
# NaN, matching the original design where St. Mary data only starts once
# Apr 1 is reached).
# ============================================
feb1_ref  = pd.Timestamp(2000, 2, 1)
apr1_ref  = pd.Timestamp(2000, 4, 1)
N_PERIODS = 16  # Feb 1 through Sep 30


def generate_semimonthly_windows(base_start, n_periods=N_PERIODS):
    """Calendar-aligned semi-monthly (1st-15th, 16th-end-of-month) window
    boundaries starting at base_start (must be day 1 or day 16)."""
    windows = []
    cur = base_start
    for i in range(n_periods):
        year, month, day = cur.year, cur.month, cur.day
        if day == 1:
            start = pd.Timestamp(year=year, month=month, day=1)
            end   = pd.Timestamp(year=year, month=month, day=15)
            nxt   = pd.Timestamp(year=year, month=month, day=16)
            label = f"{start.strftime('%b')} 1-15"
        elif day == 16:
            start = pd.Timestamp(year=year, month=month, day=16)
            end   = start + pd.offsets.MonthEnd(0)
            nxt_anchor = start + pd.DateOffset(months=1)
            nxt   = pd.Timestamp(year=nxt_anchor.year, month=nxt_anchor.month, day=1)
            label = f"{start.strftime('%b')} 16-{end.day}"
        else:
            raise ValueError(f"base_start / period anchor must fall on day 1 or day 16; got {cur}")
        windows.append((label, start, end))
        cur = nxt
    return windows


semimonthly_windows = generate_semimonthly_windows(feb1_ref, N_PERIODS)

# Index of the first period at/after each region's antecedent date
apr1_period_idx = next(i for i, (_, ws, _) in enumerate(semimonthly_windows) if ws == apr1_ref)
region_start_period_idx = {'Milk River': 0, 'St. Mary': apr1_period_idx}


# In[5]:


# Build directed river graph
riv_graph = nx.DiGraph()
for idx, row in geodata.iterrows():
    if row['maindown'] != 0:
        riv_graph.add_edge(idx, row['maindown'])

# --- Results loop: scatter plot data (fixed FULL 4-month window per region) ---
results = {}
for gauge_id in gauge_ids:
    if gauge_id not in riv_graph.nodes:
        print(f"{gauge_id} not found in river graph.")
        results[gauge_id] = None
        continue
    if gauge_id not in cout.columns or gauge_id not in swe.columns:
        results[gauge_id] = None
        continue

    region  = get_gauge_region(gauge_id)
    win     = season_windows[region]
    r_start = win['start']
    r_end   = win['end']

    cout_gauge = cout[[gauge_id]].copy()

    # --- Streamflow SUM using the region-specific FULL freshet window ---
    seasonal_sums = cout_gauge.groupby(cout_gauge.index.year).apply(
        lambda df: df.loc[
            ((df.index.month > r_start[0]) |
             ((df.index.month == r_start[0]) & (df.index.day >= r_start[1]))) &
            ((df.index.month < r_end[0]) |
             ((df.index.month == r_end[0]) & (df.index.day <= r_end[1]))),
            gauge_id
        ].sum()
    )
    seasonal_sums = seasonal_sums.to_frame(name="seasonal_sum")
    seasonal_sums["seasonal_rank"] = seasonal_sums["seasonal_sum"].rank(ascending=True).astype(int)

    # --- Antecedent SWE — date depends on region ---
    dates      = antecedent_dates[region]
    antecedent = swe[[gauge_id]].reindex(dates).rename(columns={gauge_id: "antecedent_swe"})
    antecedent["antecedent_rank"] = antecedent["antecedent_swe"].rank(ascending=True).astype(int)
    antecedent.index = antecedent.index.year

    combined = seasonal_sums.merge(antecedent, left_index=True, right_index=True, how="left")
    results[gauge_id] = combined


# ### Lead Time -- now semi-monthly (twice-monthly), R^2 (= rho^2)

# In[6]:


# --- R^2 (= rho^2, from properly-fit rank correlation) per semi-monthly
#     window (NOT cumulative) ---
r2_results = {gauge_id: [np.nan] * N_PERIODS for gauge_id in gauge_ids}

for period_idx, (label, wstart, wend) in enumerate(semimonthly_windows):
    start_md = (wstart.month, wstart.day)
    end_md   = (wend.month, wend.day)

    for gauge_id in gauge_ids:
        if gauge_id not in riv_graph.nodes or gauge_id not in cout.columns:
            continue

        region = get_gauge_region(gauge_id)

        # St. Mary has no data before its antecedent date is reached
        if period_idx < region_start_period_idx[region]:
            continue

        cout_gauge = cout[[gauge_id]].copy()
        window_sums = cout_gauge.groupby(cout_gauge.index.year).apply(
            lambda df: df.loc[
                ((df.index.month > start_md[0]) |
                 ((df.index.month == start_md[0]) & (df.index.day >= start_md[1]))) &
                ((df.index.month < end_md[0]) |
                 ((df.index.month == end_md[0]) & (df.index.day <= end_md[1]))),
                gauge_id
            ].sum()
        )
        window_sums = window_sums.to_frame(name="window_sum")

        if gauge_id not in swe.columns:
            continue
        dates      = antecedent_dates[region]
        antecedent = swe[[gauge_id]].reindex(dates).rename(columns={gauge_id: "antecedent_swe"})
        antecedent.index = antecedent.index.year

        combined_tmp = window_sums.merge(
            antecedent[["antecedent_swe"]], left_index=True, right_index=True, how="left"
        )
        valid_mask = combined_tmp[["antecedent_swe", "window_sum"]].notna().all(axis=1)
        if valid_mask.sum() > 1:
            rho, _ = spearmanr(combined_tmp.loc[valid_mask, "antecedent_swe"],
                                combined_tmp.loc[valid_mask, "window_sum"])
            # R^2 = rho^2 for a bivariate rank correlation (equivalent to
            # fitting window_sum_rank ~ antecedent_swe_rank and taking the
            # coefficient of determination). Unsigned by convention.
            r2_results[gauge_id][period_idx] = rho ** 2


# ### Combined Plot

# In[8]:


# --- Matplotlib styling ---
mpl.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'lines.linewidth': 1.2,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
    'legend.frameon': False
})

# --- Figure layout ---
column_width   = 3.5
ncols          = 3
num_gauges     = len(gauge_ids)
nrows          = int(np.ceil(num_gauges / ncols))
height_scatter = 2.8
height_r2      = 3.0
fig_width      = column_width * 2.5
fig_height     = nrows * height_scatter + height_r2

fig = plt.figure(figsize=(fig_width, fig_height))
gs  = gridspec.GridSpec(nrows=nrows + 1, ncols=ncols,
                        height_ratios=[height_scatter] * nrows + [height_r2])

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
colors = [subbasin_colors.get(str(gauge_id), '#333333') for gauge_id in gauge_ids]

# --- Scatter plots ---
scatter_axes = []
for idx, gauge_id in enumerate(gauge_ids):
    ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
    scatter_axes.append(ax)

    # Panel label "a)" once, on the top-left scatter axis only.
    if idx == 0:
        ax.text(-0.18, 1.08, 'a)', transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='bottom', ha='right')

    combined = results[gauge_id]
    if combined is None:
        ax.set_visible(False)
        continue

    # Region-specific window for this gauge
    region = get_gauge_region(gauge_id)
    win    = season_windows[region]

    x = combined["antecedent_rank"]
    y = combined["seasonal_rank"]
    ax.scatter(x, y, color=colors[idx], alpha=0.7, edgecolors='k', s=35)
    max_rank = max(x.max(), y.max())
    ax.plot([0, max_rank + 1], [0, max_rank + 1], '--', color='gray', lw=1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.set_ylim(0.5, max_rank + 0.5)
    ax.grid(True, linestyle=':', linewidth=0.6)

    # R^2 annotation (= rho^2, from properly-fit rank correlation)
    valid_mask = combined[["antecedent_rank", "seasonal_rank"]].notna().all(axis=1)
    if valid_mask.sum() > 1:
        rho, pval = spearmanr(combined.loc[valid_mask, "antecedent_rank"],
                               combined.loc[valid_mask, "seasonal_rank"])
        r2 = np.round(rho ** 2, 2)
        ax.text(x.min() + 0.1 * (x.max() - x.min()),
                y.max() - 0.1 * (y.max() - y.min()),
                f'$R^2$ = {r2}', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Title
    gauge_str = str(gauge_id).zfill(7)
    if gauge_str in hype_id_to_location:
        info  = hype_id_to_location[gauge_str]
        title = (f"{info[0]}: {info[1]} ({info[2]} km²)"
                 if len(info) == 3 else f"{info[0]}: {info[1]}")
    else:
        title = f"Gauge {gauge_id}"
    ax.set_title(title, fontsize=8)

    # Axis labels
    r_start_ts = pd.Timestamp(2000, *win['start'])
    r_end_ts   = pd.Timestamp(2000, *win['end'])
    date_label = antecedent_date_label[region]
    ax.set_xlabel(f'Upstream SWE Rank ({date_label})', fontsize=7)
    ax.set_ylabel(
        f"Streamflow Volume Rank "
        f"({r_start_ts.strftime('%b %d')}–{r_end_ts.strftime('%b %d')})",
        fontsize=7
    )

# Hide unused scatter axes
for j in range(num_gauges, nrows * ncols):
    fig.add_subplot(gs[j // ncols, j % ncols]).set_visible(False)

# --- R^2 vs semi-monthly period (bottom panel) ---
r2_ax = fig.add_subplot(gs[-1, :])
r2_ax.text(-0.06, 1.05, 'b)', transform=r2_ax.transAxes,
            fontsize=13, fontweight='bold', va='bottom', ha='right')

for idx, (gauge_id, r2_list) in enumerate(r2_results.items()):
    gauge_str = str(gauge_id).zfill(7)
    if gauge_str in hype_id_to_location:
        info  = hype_id_to_location[gauge_str]
        label = f"{info[0]}: {info[1]}"
    else:
        label = f"Gauge {gauge_id}"
    r2_ax.plot(range(len(r2_list)), r2_list,
               marker='o', markersize=4, lw=1, color=colors[idx], label=label)

# Shading: ALL semi-monthly periods contained within each region's FULL
# scatter window (since each window now spans multiple periods, not one).
milk_start_idx = next(
    i for i, (_, ws, we) in enumerate(semimonthly_windows)
    if ws >= pd.Timestamp(2000, *season_windows['Milk River']['start'])
)
milk_end_idx = next(
    i for i, (_, ws, we) in enumerate(semimonthly_windows)
    if we >= pd.Timestamp(2000, *season_windows['Milk River']['end'])
)
stmary_start_idx = next(
    i for i, (_, ws, we) in enumerate(semimonthly_windows)
    if ws >= pd.Timestamp(2000, *season_windows['St. Mary']['start'])
)
stmary_end_idx = next(
    i for i, (_, ws, we) in enumerate(semimonthly_windows)
    if we >= pd.Timestamp(2000, *season_windows['St. Mary']['end'])
)

r2_ax.axvspan(milk_start_idx   - 0.5, milk_end_idx   + 0.5,
              color='steelblue',  alpha=0.15, label='Milk River Window (Feb 1-May 31)')
r2_ax.axvspan(stmary_start_idx - 0.5, stmary_end_idx + 0.5,
              color='darkorange', alpha=0.15, label='St. Mary Window (Apr 1-Jul 31)')

r2_ax.axhline(y=0, color='black', linewidth=1.2, linestyle='-', zorder=2)

# --- x-axis: one tick per semi-monthly period ---
period_labels = [lbl for lbl, _, _ in semimonthly_windows]
r2_ax.set_xticks(range(len(period_labels)))
r2_ax.set_xticklabels(period_labels, rotation=45, ha='right', fontsize=7)

r2_ax.set_xlabel("Semi-monthly period", fontsize=8)
r2_ax.set_ylabel("R² (Antecedent SWE vs\nWindow Streamflow Volume)", fontsize=8)
r2_ax.set_ylim(0, 1)
r2_ax.set_xlim(-0.5, len(period_labels) - 0.5)
r2_ax.grid(True, linestyle=':', linewidth=0.6)
r2_ax.legend(fontsize=6, ncol=2, loc='upper right', frameon=True)
plt.tight_layout()
os.makedirs(output_path, exist_ok=True)
fig.savefig(os.path.join(output_path, 'fig3_swe_scatter.png'), dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(output_path, 'fig3_swe_scatter.pdf'), format='pdf', bbox_inches='tight')
plt.show()


# In[ ]:





# In[ ]: