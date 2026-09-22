# Publishing the repository

This release updates the existing public repository while retaining its Git
history. It must not be pushed until `RELEASE_CHECKLIST.md` is complete and the
canonical manuscript-ready handoff has been copied and verified. The target is:

```bash
https://github.com/kareem-gameel/delta-learning-target-design
```

Prepare the update from a fresh clone of that repository, replace its public
working tree with the verified candidate without deleting `.git`, inspect the
diff, commit normally on `main`, and push without force. If it remains absent,
create tag `v0.2.0-corrected` and its GitHub release only after the pushed
commit is anonymously readable.

Before publishing:

1. complete `RELEASE_CHECKLIST.md` and remove the release-candidate warning;
2. run `make verify` in a clean environment;
3. inspect `repository_manifest.csv`;
4. confirm that no raw third-party data, checkpoints, LMDBs, embeddings, or
   logs are present; and
5. confirm the GitHub and release URLs in the manuscript; and
6. create no Zenodo deposit or DOI unless separately authorized.

The MIT license covers software only. The manuscript/preprint uses the
author-selected CC BY-NC-ND 4.0 license. Bundled publication data receive no
separate reuse license. See `LICENSES.md` for the path-scoped terms.
