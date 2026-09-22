#!/usr/bin/env python3
"""Independent numerical and scope checks for the corrected SI."""
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--numerical-only', action='store_true',
                    help='Run all numerical checks without historical revision-scope assertions.')
args = parser.parse_args()

SI=Path(__file__).resolve().parents[1]
D=SI/'data/test_protocol_2026-09-22'
main=set(pd.read_csv(D/'target_registry.csv').target_id)
controls={'DFTB_ref-BoE','Delta_DFTB-ref-BoE'}
assert len(main)==13 and not main.intersection(controls)
ledger=pd.read_csv(D/'tmqm_final_ledger.csv')
qm=pd.read_csv(D/'qm9_final_ledger.csv')
metrics=pd.read_csv(D/'extended_metrics_per_seed.csv')
assert len(ledger)==82 and len(qm)==82
assert ledger.test_status.eq('complete').all()
assert qm.qm9_score_status.eq('complete').all()
assert qm.checkpoint_finality.eq('final').all()
for frame in (ledger,qm):
    assert frame.run_name.is_unique
    for model,g in frame.groupby('model'):
        assert set(g.target_id)==main|controls
        for target,r in g.groupby('target_id'):
            assert set(r.seed)==({101} if target in controls else {101,102,103})

summary=pd.read_csv(D/'test_summary.csv').set_index(['model','target_id'])
for (model,target),r in ledger.groupby(['model','target_id']):
    v=r.test_mae_hartree.to_numpy()
    s=summary.loc[model,target]
    np.testing.assert_allclose(s.mae_mean,v.mean(),rtol=0,atol=1e-13)
    assert int(s.seed_count)==len(v)
    if len(v)>1: np.testing.assert_allclose(s.mae_std,v.std(ddof=1),rtol=0,atol=1e-13)
    else: assert pd.isna(s.mae_std)

rough=pd.read_csv(D/'roughness.csv')
testlabels=pd.read_csv(D/'tmqm_test_targets.csv',dtype={'mol_id':str})
common=testlabels.dropna(subset=sorted(main))
assert len(common)==7104 and common.mol_id.is_unique and common.sid.is_unique
assert rough['split'].eq('test').all()
for split in ['train','validation']:
    other=pd.read_csv(D/f'tmqm_{split}_targets.csv',dtype={'mol_id':str})
    assert set(common.mol_id).isdisjoint(other.mol_id)
    assert set(common.sid).isdisjoint(other.sid)
metadata=json.loads((D/'roughness_metadata.json').read_text())
assert metadata['split']=='test' and len(metadata['graphs'])==4
graphs={}
for record in metadata['graphs']:
    path=D/record['graph_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['graph_sha256']
    graph=np.load(path,allow_pickle=False)
    rows=common if record['cohort']=='primary' else common.dropna(subset=sorted(controls))
    assert len(rows)==record['n_molecules']==(7104 if record['cohort']=='primary' else 7068)
    np.testing.assert_array_equal(graph['mol_id'],rows.mol_id)
    np.testing.assert_array_equal(graph['sid'],rows.sid)
    edges,weights=graph['edges'],graph['weights']
    assert len(edges)==record['n_edges'] and len(np.unique(edges,axis=0))==len(edges)
    assert (edges[:,0]<edges[:,1]).all() and edges.min()>=0 and edges.max()<len(rows)
    assert np.isfinite(weights).all() and (weights>=0).all() and (weights<=1).all() and weights.sum()>0
    graphs[record['architecture'],record['cohort']]=(rows,edges,weights)
for r in rough.itertuples():
    rows,edges,weights=graphs[r.architecture,r.cohort]
    v=rows[r.target_column].to_numpy()
    iqr=np.quantile(v,.75)-np.quantile(v,.25)
    raw=np.average((v[edges[:,0]]-v[edges[:,1]])**2,weights=weights)
    np.testing.assert_allclose(r.weighted_mean_squared_edge_difference_hartree2,raw,rtol=1e-12)
    np.testing.assert_allclose(r.target_iqr_hartree,iqr,rtol=0,atol=1e-12)
    np.testing.assert_allclose(r.d_iqr,r.weighted_mean_squared_edge_difference_hartree2/iqr**2,rtol=1e-12)
    np.testing.assert_allclose(r.d_var,raw/np.var(v),rtol=1e-12)

plot=pd.read_csv(D/'roughness_error_plot_data.csv')
corr=pd.read_csv(D/'correlations.csv')
assert not corr.dftb_included.any()
for model,r in plot.groupby('model'):
    assert set(r.target_id)==main|controls
    np.testing.assert_allclose(r.nmae,r.mae/r.target_iqr_hartree,rtol=1e-12)
    np.testing.assert_allclose(r.nmae_sd,r.mae_sd/r.target_iqr_hartree,rtol=1e-12)
    p=r[r.target_id.isin(main)]
    assert len(p)==13
    c=corr.query('model==@model and analysis=="shared_IQR"').iloc[0]
    np.testing.assert_allclose(c.r_squared,pearsonr(np.log(p.d_iqr),p.nmae).statistic**2,rtol=1e-12)
    np.testing.assert_allclose(c.kendall,kendalltau(p.d_iqr,p.nmae).statistic,rtol=1e-12)
    # Poison display-only points: the pooled result must remain unchanged.
    changed=r.copy()
    changed.loc[changed.target_id.isin(controls),['d_iqr','nmae']]=[1e100,-1e100]
    p2=changed[changed.target_id.isin(main)]
    np.testing.assert_allclose(c.r_squared,pearsonr(np.log(p2.d_iqr),p2.nmae).statistic**2,rtol=1e-12)

keys=['dataset','cohort','model','target_id','seed']
a=metrics.query('scale=="native"').set_index(keys)
b=metrics.query('scale=="DFT_parent"').set_index(keys)
for column in ['mae','rmse','bias','shifted_mae','shifted_rmse']:
    np.testing.assert_allclose(a.loc[b.index,column],b[column],rtol=1e-10,atol=1e-12)
assert set(metrics.query('dataset=="qm9" and cohort=="common13" and target_id in @main').n)=={128988}
assert set(metrics.query('dataset=="qm9" and cohort=="common15"').n)=={128978}
assert set(metrics.query('dataset=="tmqm" and cohort=="dftb_valid"').n)=={7068}

audit=pd.read_csv(D/'prediction_score_audit.csv')
assert len(audit)==164 and audit.absolute_difference.max()<1e-12
for filename,expected in json.loads((D/'bundle_sha256.json').read_text()).items():
    assert hashlib.sha256((D/filename).read_bytes()).hexdigest()==expected,filename
for path in list((SI/'tables/test_protocol_2026-09-22').glob('*.tex'))+[SI/'corrected_content.tex']:
    assert not any(byte<32 and byte not in [9,10] for byte in path.read_bytes()),path
log=(SI/'build/supplementary_information.log').read_text()
assert not any(w in log for w in ['undefined','Overfull','Underfull','LaTeX Warning'])

if args.numerical_only:
    report = dict(run_count=82, prediction_score_checks=164,
                  test_graphs_independently_rescored=4,
                  roughness_rows_checked=len(rough),
                  independent_numerical_checks='passed',
                  historical_revision_scope='not checked: subsequent manuscript edits were authorized',
                  build_warnings=0)
    (SI/'build/numerical_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0)

root=SI.parent
before=root/'revision_audit/test_protocol_2026-09-22/before'
protected=[]
for original in (before/'sections').glob('*.tex'):
    current=root/'sections'/original.name
    if original.name=='section_02_results_discussion.tex':
        marker=r'\subsection{Target design controls out-of-domain transfer}'
        latest=root/'revision_audit/section25_test_2026-09-22/before/sections'/original.name
        comparison=latest.read_text() if latest.exists() else current.read_text()
        assert original.read_text().split(marker)[1]==comparison.split(marker)[1]
    else:
        assert original.read_bytes()==current.read_bytes(),current
    protected.append(str(current.relative_to(root)))
assert (before/'main.tex').read_bytes()==(root/'main.tex').read_bytes()
for original in (before/'figures').iterdir():
    if original.name.startswith(('fig01_reference_correction.','fig02_residual_design.','fig04_roughness_nmae.')): continue
    if (root/'revision_audit/section25_test_2026-09-22/before').exists() and original.name.startswith('fig03_qm9_transfer.'): continue
    assert original.read_bytes()==(root/'figures'/original.name).read_bytes(),original
    protected.append('figures/'+original.name)
report=dict(run_count=82,prediction_score_checks=164,main_targets_per_architecture=13,
            seeds=[101,102,103],dftb_seed=101,dftb_excluded_from_pooled_statistics=True,
            test_graphs_independently_rescored=4,roughness_rows_checked=len(rough),
            main_section_2_5_scope_check='delegated to scripts/check_figure4_revision.py',protected_files=protected,build_warnings=0,
            independent_numerical_checks='passed')
(SI/'build/verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
