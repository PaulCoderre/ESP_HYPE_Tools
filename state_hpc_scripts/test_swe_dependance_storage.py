#!/usr/bin/env python
# coding: utf-8
# ## Description
# Formal test of the fill-and-spill hypothesis via an interaction
# regression, replacing the stratified High/Low comparison with a single
# model fit on the FULL sample (all ~31 years) per semi-monthly window:
#
#     rank(flow) = b0 + b1*rank(SWE)_c + b2*rank(storage)_c
#                     + b3*(rank(SWE)_c * rank(storage)_c) + eps
#
# where _c denotes rank-centered (rank minus its mean), so b1/b2 are
# interpreted as "effect at the average level of the other variable."
#
# b3 is the term that actually tests the hypothesis: does SWE's effect on
# flow change with storage level? A positive, significant b3 means SWE
# matters MORE when storage is high (consistent with fill-and-spill: snow
# can't reach the outlet if storage absorbs it first). b3 ~ 0 / not
# significant means no evidence of an interaction -- SWE and storage would
# just be additive (or independently uninformative).
#
# Values are rank-transformed (not raw) to stay consistent with the
# Spearman-based analysis used throughout this pipeline, since a linear
# model on raw magnitudes would assume linearity that hasn't been
# established.

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import statsmodels.api as sm
import os

output_path = "./window_plots/"

# ### Inputs -- same as the stratified test
geodata = pd.read_csv('../state_model/GeoData.txt', index_col=0, sep='\t')
swe  = pd.read_csv('../hist_data/timeUPSNOW_DD.txt', sep='\t', skiprows=1, index_col=0)
cilv = pd.read_csv('../hist_data/timeCILV_DD.txt', sep='\t', skiprows=1, index_col=0)
cout = pd.read_csv('~/research/SMM_Models/hype/model/v10_model/final_model/hds_v3/results/timeCOUT_DD.txt', sep='\t', skiprows=1, index_col=0)

date_start = "1985-01-01"
date_end   = "2015-12-31"

geodata['maindown'] = geodata['maindown'].astype(int)
geodata.index = geodata.index.astype(int)

swe.index  = pd.to_datetime(swe.index)
cilv.index = pd.to_datetime(cilv.index)
cout.index = pd.to_datetime(cout.index)

swe.columns  = swe.columns.astype(int)
cilv.columns = cilv.columns.astype(int)
cout.columns = cout.columns.astype(int)

swe  = swe.loc[date_start:date_end]
cilv = cilv.loc[date_start:date_end]
cout = cout.loc[date_start:date_end]

gauge_ids = [58290, 58308, 58356, 58363]
hype_id_to_location = {58290: 'FRRIB', 58308: 'MREIB', 58356: 'LDCIB', 58363: 'BTCIB'}

clim_start_year = 1985
clim_end_year   = 2015
years = list(range(clim_start_year, clim_end_year + 1))

riv_graph = nx.DiGraph()
for idx, row in geodata.iterrows():
    if row['maindown'] != 0:
        riv_graph.add_edge(idx, row['maindown'])


def generate_semimonthly_windows(base_start, n_periods):
    windows = []
    cur = base_start
    for _ in range(n_periods):
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
            raise ValueError(f"anchor must fall on day 1 or 16; got {cur}")
        windows.append((label, start, end))
        cur = nxt
    return windows


feb1_ref = pd.Timestamp(2000, 2, 1)
N_PERIODS = 16
semimonthly_windows = generate_semimonthly_windows(feb1_ref, N_PERIODS)


def get_upstream_pothole_series(gauge_id):
    if gauge_id not in riv_graph.nodes:
        return None
    upstream_segments = list(nx.ancestors(riv_graph, gauge_id)) + [gauge_id]
    cilv_filtered = cilv.loc[:, cilv.columns.isin(upstream_segments)]
    if cilv_filtered.empty:
        return None
    return cilv_filtered.sum(axis=1)


def rank_center(x):
    """Rank-transform (average rank for ties) then center on the mean rank."""
    r = rankdata(x)
    return r - r.mean()


results_by_gauge = {}

for gauge_id in gauge_ids:
    if gauge_id not in cout.columns or gauge_id not in swe.columns:
        results_by_gauge[gauge_id] = None
        continue

    storage_series = get_upstream_pothole_series(gauge_id)
    if storage_series is None:
        results_by_gauge[gauge_id] = None
        continue

    storage_by_year, swe_by_year = {}, {}
    for y in years:
        ts = pd.Timestamp(year=y, month=2, day=1)
        s_val = storage_series.get(ts, np.nan)
        w_val = swe[gauge_id].get(ts, np.nan)
        if not pd.isna(s_val):
            storage_by_year[y] = s_val
        if not pd.isna(w_val):
            swe_by_year[y] = w_val

    cout_gauge = cout[[gauge_id]].copy()
    rows = []

    for period_idx, (label, wstart, wend) in enumerate(semimonthly_windows):
        start_md = (wstart.month, wstart.day)
        end_md   = (wend.month, wend.day)

        window_sums = cout_gauge.groupby(cout_gauge.index.year).apply(
            lambda df: df.loc[
                ((df.index.month > start_md[0]) |
                 ((df.index.month == start_md[0]) & (df.index.day >= start_md[1]))) &
                ((df.index.month < end_md[0]) |
                 ((df.index.month == end_md[0]) & (df.index.day <= end_md[1]))),
                gauge_id
            ].sum()
        )

        # Assemble the full-sample dataset for this window
        yrs, swe_vals, storage_vals, flow_vals = [], [], [], []
        for y in years:
            if (y in swe_by_year and y in storage_by_year
                    and y in window_sums.index and not pd.isna(window_sums.loc[y])):
                yrs.append(y)
                swe_vals.append(swe_by_year[y])
                storage_vals.append(storage_by_year[y])
                flow_vals.append(window_sums.loc[y])

        n = len(yrs)
        if n < 10:  # need reasonable df for a 3-parameter + intercept model
            rows.append({'period_index': period_idx, 'period_label': label,
                         'b1_swe': np.nan, 'p_swe': np.nan,
                         'b2_storage': np.nan, 'p_storage': np.nan,
                         'b3_interaction': np.nan, 'p_interaction': np.nan, 'n': n})
            continue

        swe_c     = rank_center(swe_vals)
        storage_c = rank_center(storage_vals)
        flow_r    = rankdata(flow_vals)
        interaction = swe_c * storage_c

        X = np.column_stack([swe_c, storage_c, interaction])
        X = sm.add_constant(X)
        model = sm.OLS(flow_r, X).fit()

        # params/pvalues order: [const, swe_c, storage_c, interaction]
        rows.append({
            'period_index': period_idx, 'period_label': label,
            'b1_swe': model.params[1], 'p_swe': model.pvalues[1],
            'b2_storage': model.params[2], 'p_storage': model.pvalues[2],
            'b3_interaction': model.params[3], 'p_interaction': model.pvalues[3],
            'n': n,
        })

    results_by_gauge[gauge_id] = pd.DataFrame(rows)

os.makedirs(output_path, exist_ok=True)
for gauge_id, df in results_by_gauge.items():
    if df is not None:
        df.to_csv(os.path.join(output_path, f'interaction_regression_{hype_id_to_location[gauge_id]}.csv'),
                   index=False)


# ============================================
# PLOT -- interaction coefficient (b3) and its significance, per gauge
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
axes_flat = axes.flatten()

period_labels = [lbl for lbl, _, _ in semimonthly_windows]

for ax, gauge_id in zip(axes_flat, gauge_ids):
    df = results_by_gauge.get(gauge_id)
    label = hype_id_to_location.get(gauge_id, str(gauge_id))
    if df is None:
        ax.set_visible(False)
        continue

    ax.plot(df['period_index'], df['b3_interaction'], marker='o', color='#0072B2', linewidth=2)
    sig = df[df['p_interaction'] < 0.05]
    ax.scatter(sig['period_index'], sig['b3_interaction'], facecolor='#0072B2',
               edgecolor='black', linewidth=1.4, s=80, zorder=5, label='p < 0.05')

    ax.axhline(0, color='black', linewidth=1)
    ax.set_title(f'Milk River: {label}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(period_labels)))
    ax.set_xticklabels(period_labels, rotation=45, ha='right', fontsize=7)

axes_flat[0].legend(loc='upper right', fontsize=9)
fig.supylabel("Interaction coefficient b3 (SWE x storage)\non rank-transformed flow", fontsize=10)
fig.suptitle('Interaction regression: does storage level change SWE\'s effect on flow?\n'
             'flow_rank ~ SWE_rank_c + storage_rank_c + SWE_rank_c*storage_rank_c\n'
             '(positive b3 = SWE matters more when storage is high, consistent with fill-and-spill)',
             fontsize=12, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(output_path, 'fig_interaction_regression.png'), dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(output_path, 'fig_interaction_regression.pdf'), format='pdf', bbox_inches='tight')
plt.show()