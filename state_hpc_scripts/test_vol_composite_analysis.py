#!/usr/bin/env python
# coding: utf-8
# ## Description
# Composite analysis v3. Years are now stratified by the basin-appropriate
# ANTECEDENT PREDICTOR established earlier in the paper, not by a spring
# streamflow-volume proxy:
#   - St. Mary basins   : upstream SWE on Apr 1 (column already represents
#                          upstream-aggregated snow, per timeUPSNOW_DD.txt)
#   - Milk River basins : upstream prairie-pothole storage (CILV) on Feb 1,
#                          summed over the gauge's upstream network via the
#                          river graph (the earlier antecedent-vs-streamflow
#                          scatter analysis showed spring snowmelt volume
#                          itself is NOT correlated with downstream flow for
#                          Milk River, so a volume-based split there would be
#                          testing the wrong variable)
#
# Both antecedent dates coincide with the basin's ESP initialization date,
# so downstream semi-monthly windows now start immediately at that date
# (no offset needed) and run through Sep 30.
#
# For each downstream window, five distributions are built exactly as in
# composite_v2:
#   Perfect-High / ESP-High / Climatology (unconditional, all years) /
#   ESP-Low / Perfect-Low
# Distributional skill scoring is still left out -- boxplots + Mann-Whitney
# significance only.
#
# Volume convention: cout (m^3/s) * SECONDS_PER_DAY, summed over the
# window -- true volume in m^3, matching the Fig 4 boxplot script.

import os
import pandas as pd
import numpy as np
import xarray as xr
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

SECONDS_PER_DAY = 86400

# ### Basin config -- antecedent predictor, ESP init dir, and Perfect/truth
# source per gauge.
plot_configs = [
    {
        'gauge_id': 58208,
        'computed_path': '../hist_data/full/0058208.txt',
        'dir_path': './esp_init_results/apr1_start_sep30_end/58208_esp_results/',
        'region': 'St. Mary', 'antecedent_md': (4, 1),
        'title': 'St. Mary: SMRIB',
    },
    {
        'gauge_id': 58213,
        'computed_path': '../hist_data/full/0058213.txt',
        'dir_path': './esp_init_results/apr1_start_sep30_end/58213_esp_results/',
        'region': 'St. Mary', 'antecedent_md': (4, 1),
        'title': 'St. Mary: SMRBB',
    },
    {
        'gauge_id': 58290,
        'computed_path': '../hist_data/full/0058290.txt',
        'dir_path': './esp_init_results/feb1_start_sep30_end/58290_esp_results/',
        'region': 'Milk River', 'antecedent_md': (2, 1),
        'title': 'Milk River: FRRIB',
    },
    {
        'gauge_id': 58308,
        'computed_path': '../hist_data/full/0058308.txt',
        'dir_path': './esp_init_results/feb1_start_sep30_end/58308_esp_results/',
        'region': 'Milk River', 'antecedent_md': (2, 1),
        'title': 'Milk River: MREIB',
    },
    {
        'gauge_id': 58356,
        'computed_path': '../hist_data/full/0058356.txt',
        'dir_path': './esp_init_results/feb1_start_sep30_end/58356_esp_results/',
        'region': 'Milk River', 'antecedent_md': (2, 1),
        'title': 'Milk River: LDCIB',
    },
    {
        'gauge_id': 58363,
        'computed_path': '../hist_data/full/0058363.txt',
        'dir_path': './esp_init_results/feb1_start_sep30_end/58363_esp_results/',
        'region': 'Milk River', 'antecedent_md': (2, 1),
        'title': 'Milk River: BTCIB',
    },
]

clim_start_year = 1985
clim_end_year   = 2015
variable = 'cout'
output_path = './test_plots/'
SEASON_END_MONTH_DAY = (9, 30)

HIGH_PCTILE = 75
LOW_PCTILE  = 25

# ### Antecedent predictor data (loaded once, shared across basins)
geodata = pd.read_csv('../state_model/GeoData.txt', index_col=0, sep='\t')
geodata['maindown'] = geodata['maindown'].astype(int)
geodata.index = geodata.index.astype(int)

cilv = pd.read_csv('../hist_data/timeCILV_DD.txt', sep='\t', skiprows=1, index_col=0)
cilv.index = pd.to_datetime(cilv.index)
cilv.columns = cilv.columns.astype(int)

swe = pd.read_csv('../hist_data/timeUPSNOW_DD.txt', sep='\t', skiprows=1, index_col=0)
swe.index = pd.to_datetime(swe.index)
swe.columns = swe.columns.astype(int)

# River graph, needed to sum upstream pothole storage for Milk River gauges
riv_graph = nx.DiGraph()
for idx, row in geodata.iterrows():
    if row['maindown'] != 0:
        riv_graph.add_edge(idx, row['maindown'])


def generate_semimonthly_windows(start, season_end_month, season_end_day, n_periods_max=30):
    """Calendar-aligned semi-monthly windows (1st-15th, 16th-end-of-month)."""
    windows = []
    cur = start
    for _ in range(n_periods_max):
        year, month, day = cur.year, cur.month, cur.day
        if day == 1:
            wstart = pd.Timestamp(year=year, month=month, day=1)
            wend   = pd.Timestamp(year=year, month=month, day=15)
            nxt    = pd.Timestamp(year=year, month=month, day=16)
            label  = f"{wstart.strftime('%b')} 1-15"
        elif day == 16:
            wstart = pd.Timestamp(year=year, month=month, day=16)
            wend   = wstart + pd.offsets.MonthEnd(0)
            nxt_anchor = wstart + pd.DateOffset(months=1)
            nxt    = pd.Timestamp(year=nxt_anchor.year, month=nxt_anchor.month, day=1)
            label  = f"{wstart.strftime('%b')} 16-{wend.day}"
        else:
            raise ValueError(f"window anchor must fall on day 1 or 16; got {cur}")

        season_end = pd.Timestamp(year=year, month=season_end_month, day=season_end_day)
        if wend > season_end:
            break

        windows.append((label, wstart, wend))
        cur = nxt
    return windows


def compute_year_volume(sim, year, month, day_start, day_end_month, day_end_day):
    """Total volume (m^3) for one historical year over [start, end] --
    "Perfect" / climatology source."""
    start_dt = pd.Timestamp(year=year, month=month, day=day_start)
    end_dt = pd.Timestamp(year=year, month=day_end_month, day=day_end_day)
    window = sim.loc[start_dt:end_dt]
    if window.empty:
        return np.nan
    return (window[variable] * SECONDS_PER_DAY).sum()


def get_antecedent_series(region, gauge_id):
    """Return a daily Series of the basin-appropriate antecedent predictor:
    upstream SWE (St. Mary) or upstream-network pothole storage (Milk
    River), NOT yet subset to any particular date."""
    if region == 'St. Mary':
        if gauge_id not in swe.columns:
            return None
        return swe[gauge_id]
    elif region == 'Milk River':
        if gauge_id not in riv_graph.nodes:
            return None
        upstream_segments = list(nx.ancestors(riv_graph, gauge_id)) + [gauge_id]
        cilv_filtered = cilv.loc[:, cilv.columns.isin(upstream_segments)]
        if cilv_filtered.empty:
            return None
        return cilv_filtered.sum(axis=1)
    else:
        raise ValueError(f"Unknown region: {region}")


def load_esp_by_year(dir_path):
    esp_by_year = {}
    if not os.path.isdir(dir_path):
        print(f"  WARNING: ESP directory not found: {dir_path}")
        return esp_by_year
    for filename in os.listdir(dir_path):
        if not filename.endswith('.nc'):
            continue
        file_path = os.path.join(dir_path, filename)
        esp = xr.open_dataset(file_path).astype(float)
        esp['DATE'] = pd.to_datetime(esp['DATE'].values)
        base_year = pd.to_datetime(esp['DATE'][0].values).year
        esp_by_year[base_year] = esp
    return esp_by_year


def compute_esp_window_volumes(esp_ds, wstart, wend):
    ds_selected = esp_ds.sel(DATE=slice(wstart, wend))
    if ds_selected['DATE'].size == 0:
        return np.array([])
    vol_series = (ds_selected[variable] * SECONDS_PER_DAY).sum(dim='DATE').to_series()
    return vol_series.values


def run_composite_for_basin(config):
    computed_path = config['computed_path']
    dir_path = config['dir_path']
    gauge_id = config['gauge_id']
    region = config['region']
    ant_month, ant_day = config['antecedent_md']
    title = config['title']

    sim = pd.read_csv(computed_path, sep='\t', index_col=0)
    sim = sim.drop('UNITS', axis=0)
    sim.index = pd.to_datetime(sim.index, errors='coerce')
    sim[variable] = pd.to_numeric(sim[variable], errors='coerce')

    years = list(range(clim_start_year, clim_end_year + 1))

    # --- Step 1: rank years by the basin-appropriate antecedent predictor ---
    ant_series = get_antecedent_series(region, gauge_id)
    if ant_series is None:
        print(f"  No antecedent data available for {title}, skipping.")
        return None, None

    ant_values = {}
    for y in years:
        ts = pd.Timestamp(year=y, month=ant_month, day=ant_day)
        val = ant_series.get(ts, np.nan)
        if not pd.isna(val):
            ant_values[y] = val

    if len(ant_values) < 8:
        print(f"  Not enough antecedent-predictor years for {title}, skipping.")
        return None, None

    vals = np.array(list(ant_values.values()))
    high_thresh = np.percentile(vals, HIGH_PCTILE)
    low_thresh  = np.percentile(vals, LOW_PCTILE)
    high_years = sorted([y for y, v in ant_values.items() if v >= high_thresh])
    low_years  = sorted([y for y, v in ant_values.items() if v <= low_thresh])

    predictor_name = 'Apr 1 upstream SWE' if region == 'St. Mary' else 'Feb 1 upstream pothole storage'
    print(f"\n{title}")
    print(f"  Antecedent predictor: {predictor_name}")
    print(f"  High group (>{HIGH_PCTILE}th pctile, {len(high_years)} yrs): {high_years}")
    print(f"  Low group  (<{LOW_PCTILE}th pctile, {len(low_years)} yrs): {low_years}")

    # --- Step 2: downstream semi-monthly windows -- start right at the
    #     antecedent date itself (== ESP init date), through Sep 30 ---
    eval_start = pd.Timestamp(year=2001, month=ant_month, day=ant_day)
    windows = generate_semimonthly_windows(eval_start, *SEASON_END_MONTH_DAY)

    # --- Step 3: load ESP hindcasts once per basin ---
    esp_by_year = load_esp_by_year(dir_path)

    # --- Step 4: per window, build the five distributions ---
    records = []
    for label, wstart, wend in windows:
        perfect_all_years = {}
        for y in years:
            vol = compute_year_volume(sim, y, wstart.month, wstart.day, wend.month, wend.day)
            if not np.isnan(vol):
                perfect_all_years[y] = vol

        for y, vol in perfect_all_years.items():
            records.append({'window_label': label, 'year': y, 'value': vol, 'source': 'Climatology'})
            if y in high_years:
                records.append({'window_label': label, 'year': y, 'value': vol, 'source': 'Perfect-High'})
            elif y in low_years:
                records.append({'window_label': label, 'year': y, 'value': vol, 'source': 'Perfect-Low'})

        for y in high_years:
            if y not in esp_by_year:
                continue
            esp_vols = compute_esp_window_volumes(esp_by_year[y], wstart.replace(year=y), wend.replace(year=y))
            for v in esp_vols:
                records.append({'window_label': label, 'year': y, 'value': v, 'source': 'ESP-High'})

        for y in low_years:
            if y not in esp_by_year:
                continue
            esp_vols = compute_esp_window_volumes(esp_by_year[y], wstart.replace(year=y), wend.replace(year=y))
            for v in esp_vols:
                records.append({'window_label': label, 'year': y, 'value': v, 'source': 'ESP-Low'})

    df = pd.DataFrame(records)
    if df.empty:
        print(f"  No downstream window data for {title}, skipping.")
        return None, None

    os.makedirs(output_path, exist_ok=True)
    df.to_csv(os.path.join(output_path, f'composite_v3_{title.replace(": ", "_").replace(" ", "_")}.csv'), index=False)

    # --- Step 5: significance tests ---
    test_results = []
    for label in df['window_label'].unique():
        w = df[df['window_label'] == label]
        row = {'window_label': label}

        ph = w[w['source'] == 'Perfect-High']['value'].values
        pl = w[w['source'] == 'Perfect-Low']['value'].values
        row['p_perfect'] = mannwhitneyu(ph, pl, alternative='two-sided')[1] if len(ph) >= 3 and len(pl) >= 3 else np.nan

        eh = w[w['source'] == 'ESP-High']['value'].values
        el = w[w['source'] == 'ESP-Low']['value'].values
        row['p_esp'] = mannwhitneyu(eh, el, alternative='two-sided')[1] if len(eh) >= 3 and len(el) >= 3 else np.nan

        test_results.append(row)

    test_df = pd.DataFrame(test_results)
    test_df.to_csv(os.path.join(output_path, f'composite_v3_pvalues_{title.replace(": ", "_").replace(" ", "_")}.csv'), index=False)

    return df, test_df


SOURCE_STYLE = {
    'Perfect-High': dict(facecolor='#0072B2', alpha=0.9, hatch=None),
    'ESP-High':     dict(facecolor='#0072B2', alpha=0.5, hatch='//'),
    'Climatology':  dict(facecolor='#999999', alpha=0.8, hatch=None),
    'ESP-Low':      dict(facecolor='#D55E00', alpha=0.5, hatch='//'),
    'Perfect-Low':  dict(facecolor='#D55E00', alpha=0.9, hatch=None),
}
SOURCE_ORDER = ['Perfect-High', 'ESP-High', 'Climatology', 'ESP-Low', 'Perfect-Low']
SOURCE_OFFSETS = [-0.32, -0.16, 0.0, 0.16, 0.32]
BOX_WIDTH = 0.14


def plot_composite_v3(df, test_df, title, ax):
    window_order = list(dict.fromkeys(df['window_label']))
    n = len(window_order)

    local_tops = []
    for label in window_order:
        w = df[df['window_label'] == label]
        for offset, source in zip(SOURCE_OFFSETS, SOURCE_ORDER):
            vals = w[w['source'] == source]['value'].values
            pos = window_order.index(label) + offset
            if len(vals) == 0:
                continue
            style = SOURCE_STYLE[source]
            bp = ax.boxplot([vals], positions=[pos], widths=BOX_WIDTH,
                             patch_artist=True, showfliers=False,
                             medianprops=dict(color='black', linewidth=1.2))
            for patch in bp['boxes']:
                patch.set_facecolor(style['facecolor'])
                patch.set_alpha(style['alpha'])
                if style['hatch']:
                    patch.set_hatch(style['hatch'])
                    patch.set_edgecolor('black')

            q1, q3 = np.percentile(vals, [25, 75])
            iqr_ = q3 - q1
            top = min(vals.max(), q3 + 1.5 * iqr_)
            local_tops.append(top)

    panel_max = np.nanmax(local_tops) if local_tops else 1.0
    headroom = panel_max * 0.18

    if test_df is not None and not test_df.empty:
        for i, label in enumerate(window_order):
            row = test_df[test_df['window_label'] == label]
            if row.empty:
                continue
            for col, y_frac, tag in [('p_perfect', 1.0, 'Perfect'), ('p_esp', 0.92, 'ESP')]:
                p = row[col].values[0] if col in row else np.nan
                if pd.isna(p):
                    continue
                mark = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ax.text(i, panel_max + headroom * y_frac, f"{tag}: {mark}",
                        ha='center', va='bottom', fontsize=6.5)

    ax.set_xticks(range(n))
    ax.set_xticklabels(window_order, rotation=45, ha='right', fontsize=7)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('Volume [m³]', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0, top=panel_max + headroom * 1.3)


# ### Run for all basins
all_results = {}
for config in plot_configs:
    df, test_df = run_composite_for_basin(config)
    if df is not None:
        all_results[config['title']] = (df, test_df)

ncols = 2
nrows = int(np.ceil(len(all_results) / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows), constrained_layout=True)
axes_flat = axes.flatten()

for ax, (title, (df, test_df)) in zip(axes_flat, all_results.items()):
    plot_composite_v3(df, test_df, title, ax)

for ax in axes_flat[len(all_results):]:
    ax.set_visible(False)

legend_elements = [
    Patch(facecolor='#0072B2', alpha=0.9, label='Perfect (truth) -- High years'),
    Patch(facecolor='#0072B2', alpha=0.5, hatch='//', edgecolor='black', label='ESP -- High years'),
    Patch(facecolor='#999999', alpha=0.8, label='Climatology (unconditional, all years)'),
    Patch(facecolor='#D55E00', alpha=0.5, hatch='//', edgecolor='black', label='ESP -- Low years'),
    Patch(facecolor='#D55E00', alpha=0.9, label='Perfect (truth) -- Low years'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.suptitle(
    f'Composite analysis: Perfect vs. ESP vs. Climatology,\n'
    f'years split by basin-appropriate antecedent predictor '
    f'(St. Mary: Apr 1 SWE; Milk River: Feb 1 pothole storage)\n'
    f'High >{HIGH_PCTILE}th pctile, Low <{LOW_PCTILE}th pctile',
    fontsize=13, fontweight='bold')

os.makedirs(output_path, exist_ok=True)
fig.savefig(os.path.join(output_path, 'fig_composite_v3.png'), dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(output_path, 'fig_composite_v3.pdf'), format='pdf', bbox_inches='tight')
plt.show()