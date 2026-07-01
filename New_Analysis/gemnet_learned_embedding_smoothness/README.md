# GemNet Learned Embedding Smoothness

This folder supports a post-training diagnostic:

1. Export atom-level hidden states from a trained GemNet checkpoint on a held-out split.
2. Pool atom embeddings into one molecular embedding per molecule.
3. Build a Euclidean kNN graph in that learned molecular embedding space.
4. Compute target roughness on the graph and compare it with the trained model's test MAE.

This is not a pre-training predictor. It asks whether the target learned well by a particular trained GemNet is also smooth in that model's learned representation.

## Recommended Molecular Pooling

Use `mean_std` as the primary molecular embedding:

```text
mol_embedding = concat(mean_atom_embedding, std_atom_embedding)
```

For a 512-dimensional GemNet atom embedding, this gives a 1024-dimensional molecular vector. It is expressive enough to retain average chemical state and within-molecule heterogeneity, while avoiding the extra redundancy of concatenating every possible pooling statistic.

Useful sensitivity checks:

```text
mean      = 512 dimensions
sum       = 512 dimensions, intentionally size-sensitive
mean_std_sum = 1536 dimensions
```

The roughness script can use either raw embeddings or z-scored embedding dimensions. The default analysis should use `std` feature scaling for Euclidean kNN, with `raw` retained as a sensitivity check.

## Trillium Export Reminder

The older clean export wrapper used a GemNet subclass named `gemnet_t_clean_export` and wrote parquet atom embeddings with columns like:

```text
mol_id, sid, formula, atom_index, atom_count, target_value, atomic_number, atom_embedding
```

Before running export jobs, locate the wrapper on Trillium:

```bash
find /scratch/gameelka /project/rrg-ovoznyy/gameelka \
  -path '*gemnet_t_clean_export.py' -o \
  -path '*main_export_clean.py' -o \
  -path '*job_export_embeddings_clean.sh'
```

For each trained GemNet target, export the test-set atom embeddings using the target's best checkpoint and test LMDB. Keep outputs separated by target, for example:

```text
/scratch/gameelka/gemnet_embedding_exports_2026_06/dft_ref_boe/test_atomic_embeddings/
/scratch/gameelka/gemnet_embedding_exports_2026_06/delta_xtb_ref_boe/test_atomic_embeddings/
```

For current project models, prefer the manifest-driven exporter in this folder over
editing the old hardcoded bash array:

```bash
cd /scratch/gameelka/deltaML_project
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

EXPORT_ROOT=/scratch/gameelka/deltaML_project/latent_embeddings/CODEX_embeddings_analysis/trillium_clean_export
OUTROOT=/scratch/gameelka/gemnet_embedding_exports_2026_06

"$PY" New_Analysis/gemnet_learned_embedding_smoothness/export_gemnet_embeddings_manifest.py \
  --manifest "$OUTROOT/gemnet_embedding_export_manifest.csv" \
  --export-root "$EXPORT_ROOT" \
  --out-root "$OUTROOT" \
  --dataset-role test \
  --flush-every-batches 100 \
  --clean
```

## Pool Atomic Embeddings

After export, pool atom embeddings:

```bash
cd /scratch/gameelka/deltaML_project
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fairchem-conda
unset PYTHONPATH
PY="$HOME/miniforge3/envs/fairchem-conda/bin/python"

"$PY" New_Analysis/gemnet_learned_embedding_smoothness/pool_atomic_embeddings.py \
  --atom-embedding-dir /scratch/gameelka/gemnet_embedding_exports_2026_06/dft_ref_boe/test_atomic_embeddings \
  --file-glob '*.parquet' \
  --poolings mean,mean_std,sum \
  --out-dir /scratch/gameelka/gemnet_embedding_exports_2026_06/dft_ref_boe/pooled
```

## Compute Roughness

Run roughness on the pooled molecular embeddings:

```bash
"$PY" New_Analysis/gemnet_learned_embedding_smoothness/compute_embedding_roughness.py \
  --embedding-path /scratch/gameelka/gemnet_embedding_exports_2026_06/dft_ref_boe/pooled/molecular_embeddings_mean_std.parquet \
  --embedding-model-id gemnet_dft_ref_boe \
  --embedding-model-label 'GemNet(DFT_ref,BoE)' \
  --targets all \
  --k 10 \
  --scalings std,raw \
  --out-csv /scratch/gameelka/gemnet_embedding_exports_2026_06/dft_ref_boe/roughness_mean_std.csv
```

The main roughness columns are:

```text
edge_mean_squared_diff          absolute graph roughness D_edge
edge_rmse                       sqrt(D_edge), in target units
edge_avg_norm_by_population_var D_edge / Var(y)
edge_rmse_over_iqr              sqrt(D_edge) / IQR(y)
```

For multiple exported GemNet models, make a manifest:

```text
embedding_model_id,embedding_model_label,pooling,embedding_path
gemnet_dft_ref_boe,"GemNet(DFT_ref,BoE)",mean_std,/scratch/gameelka/gemnet_embedding_exports_2026_06/dft_ref_boe/pooled/molecular_embeddings_mean_std.parquet
gemnet_delta_xtb_ref_boe,"GemNet(Delta^xTB_ref,BoE)",mean_std,/scratch/gameelka/gemnet_embedding_exports_2026_06/delta_xtb_ref_boe/pooled/molecular_embeddings_mean_std.parquet
```

Then run:

```bash
"$PY" New_Analysis/gemnet_learned_embedding_smoothness/run_roughness_manifest.py \
  --manifest /scratch/gameelka/gemnet_embedding_exports_2026_06/learned_embedding_manifest.csv \
  --mol-id-filter /scratch/gameelka/gemnet_embedding_exports_2026_06/manifest_exports/shared_embedding_mol_ids.csv \
  --targets all \
  --k 10 \
  --scalings std,raw \
  --out-csv /scratch/gameelka/gemnet_embedding_exports_2026_06/learned_embedding_roughness_all.csv
```

To compare roughness with test MAE, create a small metrics manifest:

```text
embedding_model_id,trained_target_id,test_mae,test_mae_over_iqr
gemnet_dft_ref_boe,dft_ref_boe,...
gemnet_delta_xtb_ref_boe,delta_xtb_ref_boe,...
```

Then join the self-target roughness rows to the test metrics:

```bash
"$PY" New_Analysis/gemnet_learned_embedding_smoothness/merge_roughness_with_test_metrics.py \
  --roughness-csv /scratch/gameelka/gemnet_embedding_exports_2026_06/learned_embedding_roughness_all.csv \
  --model-metrics-csv /scratch/gameelka/gemnet_embedding_exports_2026_06/gemnet_test_metrics_manifest.csv \
  --self-target-only \
  --out-csv /scratch/gameelka/gemnet_embedding_exports_2026_06/learned_embedding_self_target_roughness_vs_test_mae.csv
```

## Sync Commands

Because cluster sync requires MFA, run sync commands manually. From the Mac:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  New_Analysis/gemnet_learned_embedding_smoothness/ \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/deltaML_project/New_Analysis/gemnet_learned_embedding_smoothness/
```

After Trillium export and roughness computation, pull results back:

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper

mkdir -p New_Analysis/gemnet_learned_embedding_smoothness/outputs_trillium
rsync -av --progress -e "ssh -i ~/.ssh/myKey" \
  gameelka@trillium.scinet.utoronto.ca:/scratch/gameelka/gemnet_embedding_exports_2026_06/ \
  New_Analysis/gemnet_learned_embedding_smoothness/outputs_trillium/
```
