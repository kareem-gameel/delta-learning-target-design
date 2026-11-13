#!/usr/bin/env python3
"""Figure 4: corrected multiseed QM9 DFT-parent reconstruction and DFTB control."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from generate_figure2 import COLORS, DARK, ROWS

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'si/data/test_protocol_2026-09-22'
hashes=json.loads((D/'bundle_sha256.json').read_text())
sources=['extended_metrics_per_seed.csv','qm9_final_ledger.csv']
for name in sources:
    assert hashlib.sha256((D/name).read_bytes()).hexdigest()==hashes[name]
ledger=pd.read_csv(D/'qm9_final_ledger.csv')
assert ledger.qm9_score_status.eq('complete').all() and ledger.checkpoint_finality.eq('final').all()
metrics=pd.read_csv(D/sources[0])
q=metrics.query('dataset=="qm9" and model=="gemnet" and scale=="DFT_parent"')
records=[]
for target,family,fit in ROWS:
    control=family=='DFTB'
    cohort='common15' if control else 'common13'
    selected=q.query('cohort==@cohort and target_id==@target')
    direct=q.query('cohort==@cohort and target_id=="E_ref-BoE"')
    if control:
        selected=selected.query('seed==101'); direct=direct.query('seed==101')
    assert set(selected.seed)==set(direct.seed)==({101} if control else {101,102,103})
    n=128978 if control else 128988
    assert selected.n.eq(n).all() and direct.n.eq(n).all()
    row=dict(target_id=target,family=family,fit=fit,cohort=cohort,n=n,
             seed_count=len(selected),display_only_control=control)
    for k in ['mae','shifted_mae','kendall']:
        row[k]=selected[k].mean(); row[k+'_sd']=selected[k].std(ddof=1)
        row['direct_'+k]=direct[k].mean(); row['direct_'+k+'_sd']=direct[k].std(ddof=1)
    records.append(row)
frame=pd.DataFrame(records)
frame.to_csv(ROOT/'data/fig04_qm9_transfer.csv',index=False)
q.to_csv(ROOT/'data/fig04_evaluation_metrics.csv',index=False)
(ROOT/'data/fig04_provenance.json').write_text(json.dumps(dict(
    inputs={n:hashes[n] for n in sources},scale='DFT_parent',
    primary_cohort='common13: 128988 molecules; seeds 101,102,103',
    dftb_control='common15: 128978 molecules; seed101; matched direct seed101 reference',
    aggregation='arithmetic mean of per-seed scores, not ensemble predictions',
    uncertainty='sample SD for primary routes; no DFTB uncertainty estimate',
    mean_shift='oracle offset computed separately from evaluated QM9 labels for each predictor',
    caveat='DFTB training subset differs; not pooled with primary routes',
    legacy_asset='fig03_qm9_transfer is displayed Figure 4'),indent=2)+'\n')

energy_columns = [col for col in frame if 'mae' in col]
frame[energy_columns] *= 627.5094740631

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10.5,
    'xtick.labelsize':9,'ytick.labelsize':9.5,'axes.linewidth':.8,'savefig.dpi':450,'pdf.fonttype':42})
y=np.array([8.15,7.25,5.9,5.15,3.9,3.15,1.9,1.15])
fig,axes=plt.subplots(1,3,figsize=(10.4,5.1))
configs=[('mae',(0,190),'Raw MAE (kcal/mol)','.2f'),
         ('shifted_mae',(0,51),'Mean-shifted MAE (kcal/mol)','.2f'),
         ('kendall',(.43,.95),r'Kendall correlation, $\tau$','.3f')]
for i,(ax,(metric,limits,xlabel,fmt)) in enumerate(zip(axes,configs)):
    reference=frame['direct_'+metric].iloc[0]
    for low,high in [(4.7,6.35),(.7,2.35)]:
        ax.axhspan(low,high,color='#F1F4F6',lw=0,zorder=0)
    ax.set_ylim(.45,8.75); ax.set_yticks(y); ax.tick_params(axis='y',length=0)
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x',color='#DCE3E8',lw=.65); ax.set_axisbelow(True)
    ax.axvline(reference,color=DARK,ls='--',lw=1.15)
    ax.text(-.09,1.05,'abc'[i],transform=ax.transAxes,fontsize=12,fontweight='bold')
    ax.text(.5,1.005,f'Direct GemNet = {reference:{fmt}}',transform=ax.transAxes,
            fontsize=8,color=DARK,ha='center',va='bottom')
    for yy,r in zip(y,frame.itertuples()):
        value=getattr(r,metric); ref=getattr(r,'direct_'+metric); color=COLORS[r.family]
        ax.annotate('',xy=(value,yy),xytext=(ref,yy),
                    arrowprops={'arrowstyle':'->','color':color,'lw':1.2},zorder=2)
        if r.display_only_control:
            ax.plot([ref,ref],[yy-.20,yy+.20],color=color,lw=1.3,zorder=3)
        else:
            ax.errorbar(value,yy,xerr=getattr(r,metric+'_sd'),fmt='none',ecolor=color,capsize=2,zorder=3)
        ax.plot(value,yy,ls='none',marker='D',ms=5.7,mfc='white',mec=color,mew=1.25,zorder=4)
        span=limits[1]-limits[0]; direction=-1 if value<ref else 1
        if value<limits[0]+.12*span: direction=1
        elif value>limits[1]-.12*span: direction=-1
        ax.text(value+direction*.017*span,yy+.16,format(value,fmt),color=color,
                fontsize=8.5,ha='left' if direction>0 else 'right',va='bottom')
    ax.set(xlim=limits,xlabel=xlabel)
axes[0].set_yticklabels([r'DFTB$^{\dagger}$' if r.family=='DFTB' else
    (r.family if not r.fit else f'{r.family} ({r.fit})') for r in frame.itertuples()])
axes[1].set_yticklabels([]); axes[2].set_yticklabels([])
axes[0].set_xticks([0,50,100,150]); axes[1].set_xticks([0,10,20,30,40,50]); axes[2].set_xticks([.5,.7,.9])
fig.subplots_adjust(left=.135,right=.99,top=.89,bottom=.15,wspace=.24)
for ext in ['png','pdf']:
    fig.savefig(ROOT/f'figures/fig03_qm9_transfer.{ext}',bbox_inches='tight')
plt.close(fig)
print(frame.to_string(index=False))
