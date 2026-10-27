import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# ## Description
# Rank histogram following the EXACT construction in Hamill (2001),
# Eqs. (3)-(4): for each verification (year), pool the truth with the
# sorted n-member ensemble and tally which of the n+1 discrete slots it
# falls into. This requires a CONSTANT ensemble size n across all pooled
# samples -- confirmed true here, so the discrete (not PIT-normalized)
# version is used, matching Hamill exactly rather than the continuous
# approximation used for variable-n ensembles.
#
# Rank j (1..n+1) population is the fraction of times the truth falls
# between sorted ensemble members j-1 and j. Ties (truth exactly equal to
# one or more ensemble members) are broken by randomly assigning the rank
# among the tied slots, per Hamill and Colucci (1997, 1998) -- included
# here for correctness even though ties are effectively impossible with
# continuous summed-volume values.
#
#   - Flat/uniform  -> well-calibrated spread
#   - U-shaped       -> underdispersive (ensemble spread too narrow)
#   - Hump/peaked    -> overdispersive (ensemble spread too wide)
#   - Skewed         -> systematic bias in one direction

# Inputs -- same window/config as the CRPSS boxplot script
plot_configs = [
    {
        'dir_path': './esp_init_results/apr1_start_sep30_end/58208_esp_results/',
        'computed_path': '../hist_data/full/0058208.txt',
        'start_date': '04-01',
        'end_date': '07-31',
        'title': 'St. Mary: SMRIB',
    },
    {
        'dir_path': './esp_init_results/apr1_start_sep30_end/58213_esp_results/',
        'computed_path': '../hist_data/full/0058213.txt',
        'start_date': '04-01',
        'end_date': '07-31',
        'title': 'St. Mary: SMRBB',
    },
    {
        'dir_path': './esp_init_results/feb1_start_sep30_end/58290_esp_results/',
        'computed_path': '../hist_data/full/0058290.txt',
        'start_date': '02-01',
        'end_date': '05-31',
        'title': 'Milk River: FRRIB',
    },
    {
        'dir_path': './esp_init_results/feb1_start_sep30_end/58308_esp_results/',
        'computed_path': '../hist_data/full/0058308.txt',
        'start_date': '02-01',
        'end_date': '05-31',
        'title': 'Milk River: MREIB',
    },
    {
        'dir_path': './esp_init_results/feb1_start_sep30_end/58356_esp_results/',
        'computed_path': '../hist_data/full/0058356.txt',
        'start_date': '02-01',
        'end_date': '05-31',
        'title': 'Milk River: LDCIB',
    },
    {
        'dir_path': './esp_init_results/feb1_start_sep30_end/58363_esp_results/',
        'computed_path': '../hist_data/full/0058363.txt',
        'start_date': '02-01',
        'end_date': '05-31',
        'title': 'Milk River: BTCIB',
    },
]

output_path = './new_plots/'
SECONDS_PER_DAY = 86400
RNG = np.random.default_rng(42)  # fixed seed -- reproducible tie-breaking

# Coarse-bin grouping for VISUALIZATION only -- the underlying rank
# calculation is still Hamill's exact discrete rank (unchanged). With
# only ~31 hindcast years spread across n+1=31 possible raw ranks, the
# expected count per raw bin is only ~1, which is far too little to
# distinguish a real calibration shape from sampling noise (Hamill's own
# Fig. 1 example uses 10,000 tallies per panel for a 25-member ensemble).
# Grouping adjacent raw ranks into N_COARSE_BINS bins raises the expected
# count per bin to something actually informative, at the cost of losing
# resolution on exactly where in the tail any bias sits.
N_COARSE_BINS = 6


def compute_ranks(config):
    """
    Return an array of integer ranks (1..n+1) for one gauge, one per
    hindcast year, using Hamill's exact discrete construction. Also
    returns the confirmed constant ensemble size n (raises if it is not
    actually constant across years, since the discrete construction is
    only valid when it is).
    """
    directory_path = config['dir_path']
    computed_path = config['computed_path']
    start_date = config['start_date']
    end_date = config['end_date']

    start_month, start_day = map(int, start_date.split('-'))
    end_month, end_day = map(int, end_date.split('-'))

    sim = pd.read_csv(computed_path, sep='\t', index_col=0)
    sim = sim.drop('UNITS', axis=0)
    sim.index = pd.to_datetime(sim.index, errors='coerce')
    sim['cout'] = pd.to_numeric(sim['cout'], errors='coerce')

    ranks = []
    ensemble_sizes = set()

    if not os.path.isdir(directory_path):
        return np.array(ranks), None

    for filename in os.listdir(directory_path):
        if not filename.endswith('.nc'):
            continue
        file_path = os.path.join(directory_path, filename)
        esp = xr.open_dataset(file_path).astype(float)
        esp['DATE'] = pd.to_datetime(esp['DATE'])

        last_year = pd.to_datetime(esp['DATE'][-1].values).year
        start_dt = pd.Timestamp(year=last_year, month=start_month, day=start_day)
        end_dt = pd.Timestamp(year=last_year, month=end_month, day=end_day)

        ds_selected = esp.sel(DATE=slice(start_dt, end_dt))
        ensemble_vals = (ds_selected['cout'] * SECONDS_PER_DAY).sum(dim='DATE').values
        ensemble_vals = ensemble_vals[~np.isnan(ensemble_vals)]
        n = len(ensemble_vals)
        if n < 2:
            continue
        ensemble_sizes.add(n)

        model_sim = sim.loc[start_dt:end_dt]
        obs_val = (model_sim['cout'] * SECONDS_PER_DAY).sum()
        if pd.isna(obs_val):
            continue

        # Hamill Eq. (3)-(4): rank j is where the truth falls when pooled
        # with the sorted ensemble. less = number of ensemble members
        # strictly below the truth; equal = number exactly equal (tie
        # group). Deterministic rank = less + 1 when no ties; with ties,
        # randomly assign among the (equal + 1) slots the tie spans, per
        # Hamill and Colucci's randomized tie-breaking rule.
        less = int(np.sum(ensemble_vals < obs_val))
        equal = int(np.sum(ensemble_vals == obs_val))
        if equal == 0:
            rank = less + 1
        else:
            rank = less + 1 + RNG.integers(0, equal + 1)
        ranks.append(rank)

    if len(ensemble_sizes) > 1:
        raise ValueError(
            f"Ensemble size is NOT constant for {config['title']}: "
            f"found sizes {sorted(ensemble_sizes)}. Hamill's discrete rank "
            f"histogram requires a constant n across all pooled samples."
        )

    n_const = ensemble_sizes.pop() if ensemble_sizes else None
    return np.array(ranks), n_const


# ============================================
# PLOT -- one rank histogram per gauge, n+1 discrete bins, 2x3 grid
# ============================================
ncols = 2
nrows = int(np.ceil(len(plot_configs) / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4.0 * nrows), constrained_layout=True)
axes_flat = axes.flatten()

for ax, config in zip(axes_flat, plot_configs):
    ranks, n_const = compute_ranks(config)
    title = config['title']

    if len(ranks) == 0 or n_const is None:
        ax.set_title(f"{title}\n(no data)", fontsize=10)
        ax.set_visible(False)
        continue

    n_bins = n_const + 1  # Hamill: n+1 possible discrete raw ranks

    # --- Group the raw ranks (1..n_bins) into N_COARSE_BINS bins for
    #     plotting, so each bin has enough expected counts to be
    #     informative. The rank values themselves (`ranks`) are untouched
    #     -- this only changes how they're tallied for the figure. ---
    coarse_edges = np.linspace(1, n_bins + 1, N_COARSE_BINS + 1)
    counts, _ = np.histogram(ranks, bins=coarse_edges)
    freqs = counts / counts.sum()
    expected_freq = 1.0 / N_COARSE_BINS

    bin_widths = np.diff(coarse_edges)
    bin_centers = coarse_edges[:-1] + bin_widths / 2

    ax.bar(bin_centers, freqs, width=bin_widths * 0.9, align='center',
           color='steelblue', edgecolor='black', alpha=0.85)
    ax.axhline(expected_freq, color='red', linestyle='--', linewidth=1.5,
               label='Uniform (ideal)')

    # x-tick labels show the raw-rank RANGE each coarse bin covers, so the
    # figure still reads as "low ranks / mid ranks / high ranks" rather
    # than losing all reference to the original 1..n_bins scale.
    tick_labels = [f"{int(round(coarse_edges[i]))}-{int(round(coarse_edges[i+1]) - 1)}"
                   for i in range(N_COARSE_BINS)]

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f'Rank group (of 1-{n_bins}, {N_COARSE_BINS} bins)', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlim(coarse_edges[0], coarse_edges[-1])
    ax.set_ylim(0, max(freqs.max(), expected_freq) * 1.3)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.text(0.02, 0.95, f'n_years = {len(ranks)}\nensemble size = {n_const}\n'
            f'~{n_bins / N_COARSE_BINS:.1f} raw ranks/bin, '
            f'~{len(ranks) / N_COARSE_BINS:.1f} counts/bin expected',
            transform=ax.transAxes, fontsize=7, va='top', ha='left')

for ax in axes_flat[len(plot_configs):]:
    ax.set_visible(False)

fig.suptitle('Rank Histograms (Hamill 2001, coarse-binned) -- ESP Ensemble Calibration Check\n'
             '(flat = well-calibrated; U-shape = underdispersive; hump = overdispersive; skew = biased)',
             fontsize=12, fontweight='bold')

os.makedirs(output_path, exist_ok=True)
fig.savefig(os.path.join(output_path, 'fig_rank_histograms_hamill.png'), dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(output_path, 'fig_rank_histograms_hamill.pdf'), format='pdf', bbox_inches='tight')
plt.show()