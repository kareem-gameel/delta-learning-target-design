#!/usr/bin/env python3
"""Figure 3: corrected test roughness versus normalized GemNet test error."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'si/data/test_protocol_2026-09-22'
INPUTS = ['roughness_error_plot_data.csv', 'correlations.csv', 'target_registry.csv', 'tmqm_final_ledger.csv']
hashes = json.loads((BUNDLE/'bundle_sha256.json').read_text())
for name in INPUTS:
    assert hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest() == hashes[name], name
frame = pd.read_csv(BUNDLE/INPUTS[0]).query('model=="gemnet"').set_index('target_id')
primary = pd.read_csv(BUNDLE/'target_registry.csv').target_id.tolist()
controls = ['DFTB_ref-BoE', 'Delta_DFTB-ref-BoE']
assert len(primary) == 13 and set(frame.index) == set(primary + controls)
assert frame['split'].eq('test').all()
np.testing.assert_allclose(frame.nmae, frame.mae/frame.target_iqr_hartree)
np.testing.assert_allclose(frame.nmae_sd, frame.mae_sd/frame.target_iqr_hartree)
fit = frame.loc[primary]
assert not fit.index.isin(controls).any() and fit.n_molecules.eq(7104).all()
assert frame.loc[controls,'n_molecules'].eq(7068).all()
slope, intercept = np.polyfit(np.log(fit.d_iqr), fit.nmae, 1)
r = pearsonr(np.log(fit.d_iqr), fit.nmae).statistic
expected = pd.read_csv(BUNDLE/'correlations.csv').query('model=="gemnet" and analysis=="shared_IQR"').iloc[0]
np.testing.assert_allclose([r*r, slope, intercept], [expected.r_squared, expected.slope, expected.intercept])
frame['included_in_fit'] = frame.index.isin(primary)
frame.to_csv(ROOT/'data/fig03_test_roughness.csv')
provenance = dict(figure_number=3, asset_path='figures/fig04_roughness_nmae',
    inputs={n:dict(path=str((BUNDLE/n).relative_to(ROOT)),sha256=hashes[n]) for n in INPUTS},
    split='test', normalization='test MAE / same-target test IQR',
    fitting_target_ids=primary, display_only_target_ids=controls,
    n_fitting_targets=13, n_displayed_targets=15,
    r_squared=float(r*r), pearson=float(r),
    spearman=float(spearmanr(fit.d_iqr,fit.nmae).statistic),
    kendall=float(kendalltau(fit.d_iqr,fit.nmae).statistic), slope=float(slope), intercept=float(intercept),
    uncertainty='Sample SD retained in source CSV but not plotted; primary points are means of seeds 101, 102, and 103; DFTB has seed 101 only.',
    interpretation='Descriptive in-sample cross-target fit, not a prospective performance guarantee.')
(ROOT/'data/fig03_test_roughness_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

styles = [
    ('E_ref', r'$E_{\rm ref}$', '#4B5563', 'o', (10,-14)),
    ('xTB_ref', r'$E_{\rm xTB,ref}$', '#168C9E', 'o', (-62,-14)),
    ('E_ref-BoE', r'$E_{\rm ref,BoE}$', '#2E6FDE', 'o', (8,25)),
    ('Delta_MLP_BoE-ref', r'$E_{\rm ref,BoE}^{\rm (MLP)}$', '#7A5195', 'P', (10,8)),
    ('xTB_ref-BoE', r'$E_{\rm xTB,ref,BoE}$', '#168C9E', 'o', (-90,4)),
    ('Delta_xTB-ref', r'$\Delta^{\rm xTB}_{\rm ref}$', '#00876C', 'D', (-4,22)),
    ('Delta_xTB-ref-BoE', r'$\Delta^{\rm xTB}_{\rm ref,BoE}$', '#00876C', 'D', (-77,32)),
    ('Delta_BoB-ref-BoE', r'$\Delta^{\rm BoB}_{\rm lin}$', '#3B75AF', 's', (-48,14)),
    ('Delta_PRE-ref-BoE', r'$\Delta^{\rm PRE}_{\rm lin}$', '#E59A18', 's', (9,-12)),
    ('Delta_PRE_plus_A-ref-BoE', r'$\Delta^{\rm PRE+A}_{\rm lin}$', '#A45C9C', 's', (9,10)),
    ('Delta_MLP_BoB-ref-BoE', r'$\Delta^{\rm BoB}_{\rm MLP}$', '#D46A1F', '^', (9,13)),
    ('Delta_MLP_PRE-ref-BoE', r'$\Delta^{\rm PRE}_{\rm MLP}$', '#4B9E42', '^', (9,10)),
    ('Delta_MLP_PRE_plus_A-ref-BoE', r'$\Delta^{\rm PRE+A}_{\rm MLP}$', '#B36B9E', '^', (-10,14)),
    ('DFTB_ref-BoE', r'$E_{\rm DFTB,ref,BoE}^{\dagger}$', '#B52B79', 'o', (-25,-24)),
    ('Delta_DFTB-ref-BoE', r'$\Delta^{\rm DFTB\dagger}_{\rm ref,BoE}$', '#B52B79', 'D', (37,7)),
]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10.5,
                     'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.8,
                     'savefig.dpi':450,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.6,4.75))
xline=np.geomspace(fit.d_iqr.min()*.88,fit.d_iqr.max()*1.12,300)
ax.plot(xline,intercept+slope*np.log(xline),color='#6B7280',ls='--',lw=1.25,zorder=1)
for target,label,color,marker,offset in styles:
    row=frame.loc[target]
    ax.scatter(row.d_iqr,row.nmae,s=80 if marker!='o' else 68,marker=marker,
               color=color,edgecolor='#20262D',lw=.8,zorder=3)
    ax.annotate(label,(row.d_iqr,row.nmae),xytext=offset,textcoords='offset points',
                fontsize=8.4,color=color,ha='left',va='center',
                bbox={'facecolor':'white','edgecolor':'none','pad':.4} if target=='Delta_MLP_BoE-ref' else None,
                arrowprops={'arrowstyle':'-','color':color,'lw':.55} if row.nmae<.15 else None)
ax.set_xscale('log')
ax.set(xlabel=r'Normalized test roughness, $D_{\rm IQR}$',
       ylabel=r'GemNet normalized test error, $n$-MAE',xlim=(.29,19),ylim=(-.065,1.08))
ax.spines[['top','right']].set_visible(False)
ax.grid(True,color='#DCE3E8',linewidth=.75)
ax.set_axisbelow(True)
ax.text(.03,.96,'semi-log fit\n'+rf'$R^2={r*r:.3f}$',transform=ax.transAxes,
        ha='left',va='top',fontsize=9.2,bbox={'facecolor':'white','edgecolor':'#B8C2CC','pad':4})
handles=[Line2D([],[],marker=m,ls='none',mfc=c,mec='#20262D',label=label) for m,c,label in [
    ('o','white','full/baseline targets'),('P','#7A5195','nonlinear BoE residual'),
    ('D','#00876C','xTB residuals'),('s','#3B75AF','linear residuals'),
    ('^','#D46A1F','MLP residuals'),('D','#B52B79',r'DFTB$^{\dagger}$ controls (excluded)')]]
ax.legend(handles=handles,frameon=False,loc='lower right',ncol=2,columnspacing=.9,
          handletextpad=.35,fontsize=8)
fig.subplots_adjust(left=.13,right=.98,top=.97,bottom=.14)
for ext in ['png','pdf']:
    fig.savefig(ROOT/f'figures/fig04_roughness_nmae.{ext}',bbox_inches='tight')
plt.close(fig)
print(json.dumps({k:provenance[k] for k in ['r_squared','pearson','spearman','kendall','slope','intercept','n_fitting_targets','n_displayed_targets']},indent=2))
