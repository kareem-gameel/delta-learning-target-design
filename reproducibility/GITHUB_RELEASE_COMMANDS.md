# GitHub Release Commands

Use these commands when the repository is ready to publish. They are not run
automatically because authentication and remote synchronization may require
manual approval.

## 1. Regenerate the Manifest

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper
python3 reproducibility/build_artifact_manifest.py
```

## 2. Create the Artifact Zip

```bash
cd /Users/kareem/PycharmProjects/deltaML_Paper
python3 reproducibility/make_release_bundle.py
```

This creates:

```text
reproducibility/deltaml_paper_release_artifacts.zip
```

## 3. Commit the Clean Release Files

Review `git status` carefully before committing. The current working tree has
legacy deleted files and many local generated artifacts, so a clean release
branch or clean repo is preferred.

Suggested minimal add command:

```bash
git add README.md RUN_REPRODUCTION.md DATA_AND_CODE_AVAILABILITY_PLAN.md \
  MODEL_METHODOLOGY_REPORT.md environment.yml .zenodo.json .gitignore \
  reproducibility manuscript New_Analysis Target_Design scripts \
  trillium_project/split_ids
```

If `.gitignore` blocks a small manuscript table or figure that should be
tracked, add it explicitly with `git add -f <path>`.

## 4. Tag and Push

```bash
git tag -a v0.1-paper-artifacts -m "Paper reproducibility release"
git push origin HEAD
git push origin v0.1-paper-artifacts
```

## 5. Create the GitHub Release

If GitHub CLI is available:

```bash
gh release create v0.1-paper-artifacts \
  reproducibility/deltaml_paper_release_artifacts.zip \
  --title "DeltaML paper reproducibility artifacts v0.1" \
  --notes "Processed targets, split IDs, figure/table source data, and analysis manifests for the manuscript."
```

If GitHub CLI is not available, create the release in the GitHub web UI and
upload `reproducibility/deltaml_paper_release_artifacts.zip` manually.

## 6. Optional DOI

If the journal requires a DOI, upload the same zip to Zenodo, Figshare, OSF, or
Dataverse and cite the DOI in `manuscript/main.tex`.

