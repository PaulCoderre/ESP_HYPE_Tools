#!/usr/bin/env python
# coding: utf-8
# ## Description
# Before interpreting pothole storage's higher R^2 (vs. SWE, vs. soil
# moisture) as evidence it captures something the other two don't, check
# whether the three Feb 1 antecedent states are themselves correlated
# across years. In the model, all soil column outflow enters the pothole
# complex, so some correlation is expected by construction -- the
# question is how much, and whether it's strong enough that storage's
# apparent predictive edge could just be it acting as a cleaner, less
# noisy integrator of the same underlying winter signal, rather than
# carrying genuinely independent information.
#
# For each Milk River gauge: pull Feb 1 SWE, Feb 1 soil moisture, and
# Feb 1 pothole storage (upstream-network-summed) for every year, and
# compute all three pairwise Spearman correlations.

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

output_path = "./window_plots/"

# ### Inputs
geodata = pd.read_csv('../state_model/GeoData.txt', index_col=0, sep='\t')
swe  = pd.read_csv('../hist_data/timeUPSNOW_DD.txt', sep='\t', skiprows=1, index_col=0)
soim = pd.read_csv('../hist_data/timeUPSOIM_DD.txt', sep='\t', skiprows=1, index_col=0)
cilv = pd.read_csv('../hist_data/timeCILV_DD.txt', sep='\t', skiprows=1, index_col=0)

date_start = "1985-01-01"
date_end   = "2015-12-31"

geodata['maindown'] = geodata['maindown'].astype(int)
geodata.index = geodata.index.astype(int)

swe.index  = pd.to_datetime(swe.index)
soim.index = pd.to_datetime(soim.index)
cilv.index = pd.to_datetime(cilv.index)

swe.columns  = swe.columns.astype(int)
soim.columns = soim.columns.astype(int)
cilv.columns = cilv.columns.astype(int)

swe  = swe.loc[date_start:date_end]
soim = soim.loc[date_start:date_end]
cilv = cilv.loc[date_start:date_end]

gauge_ids = [58290, 58308, 58356, 58363]
hype_id_to_location = {58290: 'FRRIB', 58308: 'MREIB', 58356: 'LDCIB', 58363: 'BTCIB'}

clim_start_year = 1985
clim_end_year   = 2015
years = list(range(clim_start_year, clim_end_year + 1))

riv_graph = nx.DiGraph()
for idx, row in geodata.iterrows():
    if row['maindown'] != 0:
        riv_graph.add_edge(idx, row['maindown'])


def get_upstream_pothole_series(gauge_id):
    if gauge_id not in riv_graph.nodes:
        return None
    upstream_segments = list(nx.ancestors(riv_graph, gauge_id)) + [gauge_id]
    cilv_filtered = cilv.loc[:, cilv.columns.isin(upstream_segments)]
    if cilv_filtered.empty:
        return None
    return cilv_filtered.sum(axis=1)


# ### Pull Feb 1 values for all three antecedent states, per gauge/year
pair_results = []
scatter_data = {}

for gauge_id in gauge_ids:
    label = hype_id_to_location[gauge_id]
    storage_series = get_upstream_pothole_series(gauge_id)
    if storage_series is None or gauge_id not in swe.columns or gauge_id not in soim.columns:
        print(f"  Skipping {label}: missing data")
        continue

    swe_vals, soim_vals, storage_vals = [], [], []
    for y in years:
        ts = pd.Timestamp(year=y, month=2, day=1)
        s_val   = storage_series.get(ts, np.nan)
        w_val   = swe[gauge_id].get(ts, np.nan)
        m_val   = soim[gauge_id].get(ts, np.nan)
        if not (pd.isna(s_val) or pd.isna(w_val) or pd.isna(m_val)):
            swe_vals.append(w_val)
            soim_vals.append(m_val)
            storage_vals.append(s_val)

    n = len(swe_vals)
    if n < 8:
        print(f"  Skipping {label}: only {n} complete years")
        continue

    scatter_data[label] = {'swe': np.array(swe_vals), 'soim': np.array(soim_vals),
                            'storage': np.array(storage_vals)}

    pairs = [
        ('SWE', 'Soil Moisture', swe_vals, soim_vals),
        ('SWE', 'Pothole Storage', swe_vals, storage_vals),
        ('Soil Moisture', 'Pothole Storage', soim_vals, storage_vals),
    ]
    for var_a, var_b, a_vals, b_vals in pairs:
        rho, pval = spearmanr(a_vals, b_vals)
        pair_results.append({
            'gauge': label, 'var_a': var_a, 'var_b': var_b,
            'rho': rho, 'p_value': pval, 'n': n,
        })

results_df = pd.DataFrame(pair_results)
os.makedirs(output_path, exist_ok=True)
results_df.to_csv(os.path.join(output_path, 'antecedent_predictor_crosscorrelation.csv'), index=False)

print("\nPairwise Spearman correlations among Feb 1 antecedent states (Milk River):")
print(results_df.to_string(index=False))


# ============================================
# PLOT -- one row per gauge, three pairwise scatter panels
# ============================================
pairs_to_plot = [
    ('swe', 'soim', 'SWE', 'Soil Moisture'),
    ('swe', 'storage', 'SWE', 'Pothole Storage'),
    ('soim', 'storage', 'Soil Moisture', 'Pothole Storage'),
]

gauges_with_data = list(scatter_data.keys())
fig, axes = plt.subplots(len(gauges_with_data), 3, figsize=(13, 3.3 * len(gauges_with_data)))

for row, gauge_label in enumerate(gauges_with_data):
    d = scatter_data[gauge_label]
    for col, (key_a, key_b, name_a, name_b) in enumerate(pairs_to_plot):
        ax = axes[row, col] if len(gauges_with_data) > 1 else axes[col]
        rho, pval = spearmanr(d[key_a], d[key_b])

        ax.scatter(pd.Series(d[key_a]).rank(), pd.Series(d[key_b]).rank(),
                   alpha=0.7, edgecolor='k', s=40)
        ax.plot([1, len(d[key_a])], [1, len(d[key_a])], '--', color='gray', lw=1)
        ax.set_title(f'{gauge_label}: {name_a} vs {name_b}\n'
                     f'$\\rho$={rho:.2f}, p={pval:.3f}', fontsize=9)
        ax.set_xlabel(f'{name_a} rank (Feb 1)', fontsize=8)
        ax.set_ylabel(f'{name_b} rank (Feb 1)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

plt.tight_layout()
fig.savefig(os.path.join(output_path, 'fig3b_antecedent_crosscorrelation.png'), dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(output_path, 'fig3b_antecedent_crosscorrelation.pdf'), format='pdf', bbox_inches='tight')
plt.show()