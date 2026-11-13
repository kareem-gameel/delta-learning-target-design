#!/usr/bin/env python3
"""Generate Figure 1 from final held-out test scores and matching test IQRs."""

import csv
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    bundle = ROOT / 'si/data/test_protocol_2026-09-22'
    inputs = ['tmqm_final_ledger.csv', 'target_statistics.csv', 'tmqm_test_targets.csv']
    hashes = json.loads((bundle / 'bundle_sha256.json').read_text())
    for name in inputs:
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == hashes[name]
    ledger = pd.read_csv(bundle / inputs[0])
    statistics = pd.read_csv(bundle / inputs[1]).query('split=="test"').set_index('target_id')
    targets = pd.read_csv(bundle / inputs[2])
    rows = []
    per_seed = []
    for target in ['E_ref', 'E_ref-BoE', 'Delta_MLP_BoE-ref']:
        runs = ledger.query('model=="gemnet" and target_id==@target').sort_values('seed')
        assert runs.seed.tolist() == [101, 102, 103]
        assert runs.test_status.eq('complete').all() and runs.test_n.eq(7104).all()
        mae = runs.test_mae_hartree.to_numpy()
        iqr = statistics.loc[target, 'iqr']
        np.testing.assert_allclose(iqr, np.subtract(*np.percentile(targets[target], [75, 25])))
        assert iqr > 0 and np.isfinite(mae).all()
        rows.append({'target': target, 'n_seeds': len(runs),
                     'test_rows': 7104, 'evaluation_split': 'test', 'target_iqr_Ha': iqr,
                     'mae_mean_Ha': float(mae.mean()),
                     'mae_sample_sd_Ha': float(mae.std(ddof=1)),
                     'nmae_mean': float(mae.mean() / iqr),
                     'nmae_sample_sd': float(mae.std(ddof=1) / iqr)})
        per_seed.extend({'target': target, 'seed': int(r.seed), 'test_rows': 7104,
                         'test_mae_Ha': r.test_mae_hartree, 'target_iqr_Ha': iqr,
                         'test_nmae': r.test_mae_hartree / iqr,
                         'best_checkpoint_path': r.best_checkpoint_path,
                         'best_checkpoint_sha256': r.best_checkpoint_sha256} for r in runs.itertuples())
    (ROOT / 'data/fig01_provenance.json').write_text(json.dumps({
        'bundle': str(bundle.relative_to(ROOT)), 'inputs_sha256': {n: hashes[n] for n in inputs},
        'evaluation_split': 'test', 'normalization': 'same-target test IQR',
        'checkpoint_selection': 'validation only', 'uncertainty': 'sample SD across three GNN seeds',
    }, indent=2) + '\n')
    write_csv(ROOT / 'data/fig01_reference_correction.csv', rows)
    write_csv(ROOT / 'data/fig01_per_seed.csv', per_seed)
    parent, linear, mlp = rows
    ratios = {
        'linear_mae_improvement': parent['mae_mean_Ha'] / linear['mae_mean_Ha'],
        'linear_iqr_shrinkage': parent['target_iqr_Ha'] / linear['target_iqr_Ha'],
        'linear_nmae_increase': linear['nmae_mean'] / parent['nmae_mean'],
        'mlp_vs_linear_iqr_shrinkage': linear['target_iqr_Ha'] / mlp['target_iqr_Ha'],
        'mlp_vs_linear_nmae_increase': mlp['nmae_mean'] / linear['nmae_mean'],
        'mlp_vs_linear_mae_increase': mlp['mae_mean_Ha'] / linear['mae_mean_Ha'],
    }
    (ROOT / 'data/fig01_ratios.json').write_text(json.dumps(ratios, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5,
                         'axes.labelsize': 10, 'xtick.labelsize': 8.5,
                         'ytick.labelsize': 8.5, 'axes.linewidth': .8,
                         'figure.dpi': 180, 'savefig.dpi': 450,
                         'pdf.fonttype': 42})
    labels = [r'$E_{\rm ref}$', r'$E_{\rm ref,BoE}^{(\rm linear)}$',
              r'$E_{\rm ref,BoE}^{(\rm MLP)}$']
    colors = ['#57789B', '#C65D3A', '#7A5195']
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15))
    x = np.arange(3)
    panels = [
        ('target_iqr_Ha', None, 'Target IQR (kcal/mol)', (25, 910), '.1f'),
        ('mae_mean_Ha', 'mae_sample_sd_Ha', 'GemNet test MAE (kcal/mol)', (0, 16.4), '.2f'),
        ('nmae_mean', 'nmae_sample_sd', r'GemNet test $n$-MAE', (0, .43), '.3f'),
    ]
    for index, (ax, (key, sd_key, ylabel, ylim, fmt)) in enumerate(zip(axes, panels)):
        values = np.array([r[key] for r in rows])
        errors = np.array([r[sd_key] for r in rows]) if sd_key else np.zeros(3)
        if key.endswith("_Ha"):
            values *= 627.5094740631
            errors *= 627.5094740631
        ax.bar(x, values, width=.62, color=colors, edgecolor='#1F2933', linewidth=.7)
        if sd_key:
            ax.errorbar(x, values, yerr=errors, fmt='none', ecolor='#1F2933',
                        capsize=3.5, linewidth=1, zorder=4)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        if index == 0:
            ax.set_yscale('log')
        ax.set_ylim(*ylim)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#DCE3E8', linewidth=.75)
        ax.set_axisbelow(True)
        for position, value, error in zip(x, values, errors):
            offset = value * .06 if index == 0 else (ylim[1] - ylim[0]) * .025
            ax.text(position, value + error + offset, format(value, fmt),
                    ha='center', va='bottom', fontsize=8.1)
        ax.text(-.18, 1.04, chr(ord('a') + index), transform=ax.transAxes,
                ha='left', va='bottom', fontsize=12, fontweight='bold')
    fig.subplots_adjust(left=.075, right=.99, top=.92, bottom=.20, wspace=.42)
    for suffix in ('png', 'pdf'):
        fig.savefig(ROOT / f'figures/fig01_reference_correction.{suffix}', bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'summary': rows, 'ratios': ratios}, indent=2))


if __name__ == '__main__':
    main()
