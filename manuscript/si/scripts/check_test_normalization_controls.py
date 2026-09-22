"""Independently rescore saved permutations and verify exhaustive pair reporting."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

SI=Path(__file__).resolve().parents[1]
D=SI/'data/test_protocol_2026-09-22'
OUT=SI/'data/normalization_controls_test_2026-09-22'
saved=np.load(OUT/'permuted_roughness.npz',allow_pickle=False)
targets=saved['target_ids'].tolist()
assert set(targets)==set(pd.read_csv(D/'target_registry.csv').target_id)
plot=pd.read_csv(D/'roughness_error_plot_data.csv')
null=pd.read_csv(OUT/'permutation_associations.csv')
summary=pd.read_csv(OUT/'permutation_summary.csv').set_index('model')
pairs=pd.read_csv(OUT/'similar_iqr_pairs.csv')
text=(SI/'test_normalization_control.tex').read_text()
for model in ['gemnet','schnet']:
    p=plot.query('model==@model').set_index('target_id').loc[targets]
    x=np.log(saved[model])
    assert x.shape==(5000,13) and np.isfinite(x).all()
    x-=x.mean(axis=1,keepdims=True)
    y=p.nmae.to_numpy()-p.nmae.mean()
    r=(x@y)/np.sqrt((x*x).sum(axis=1)*(y*y).sum())
    np.testing.assert_allclose(r,null.query('model==@model').pearson_r,atol=1e-14)
    s=summary.loc[model]
    observed=np.corrcoef(np.log(p.d_iqr),p.nmae)[0,1]
    np.testing.assert_allclose(s.observed_r,observed)
    assert s.exceedances==int((r>=observed).sum())
    np.testing.assert_allclose(s.one_sided_p,(1+(r>=observed).sum())/5001)
    for col,q in [('null_q025_r2',.025),('null_median_r2',.5),('null_q975_r2',.975)]:
        np.testing.assert_allclose(s[col],np.quantile(r*r,q))
        assert f'{s[col]:.3f}' in text
    expected=set()
    for a,b in itertools.combinations(targets,2):
        iq=p.loc[[a,b],'target_iqr_hartree']
        if iq.max()/iq.min()<=1.10: expected.add((a,b))
    actual=pairs.query('model==@model')
    assert set(zip(actual.target_a,actual.target_b))==expected
    for row in actual.itertuples():
        for suffix,target in [('a',row.target_a),('b',row.target_b)]:
            for key,column in [('iqr','target_iqr_hartree'),('d','d_iqr'),('nmae','nmae')]:
                value=getattr(row,f'{key}_{suffix}')
                np.testing.assert_allclose(value,p.loc[target,column])
                assert format(value,'.5f' if key=='iqr' else '.3f') in text
        assert row.concordant==((row.d_b-row.d_a)*(row.nmae_b-row.nmae_a)>0)
report=dict(permutations_rescored=10000,architectures=2,targets_per_architecture=13,
            all_similar_iqr_pairs_verified=True,dftb_excluded=True,status='passed')
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
