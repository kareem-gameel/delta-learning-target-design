from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ReleaseDataTests(unittest.TestCase):
    @property
    def corrected(self) -> Path:
        return ROOT / "manuscript" / "si" / "data" / "test_protocol_2026-09-22"

    def test_splits_are_unique_and_disjoint(self) -> None:
        split_ids = {}
        expected_counts = {"train": 56830, "val": 7104, "test": 7104}
        for split, expected_count in expected_counts.items():
            frame = pd.read_csv(ROOT / "splits" / f"tmqm_{split}_ids.csv", usecols=["mol_id"])
            self.assertEqual(len(frame), expected_count)
            self.assertFalse(frame["mol_id"].isna().any())
            self.assertFalse(frame["mol_id"].duplicated().any())
            split_ids[split] = set(frame["mol_id"].astype(str))
        self.assertTrue(split_ids["train"].isdisjoint(split_ids["val"]))
        self.assertTrue(split_ids["train"].isdisjoint(split_ids["test"]))
        self.assertTrue(split_ids["val"].isdisjoint(split_ids["test"]))

    def test_corrected_ledgers_are_complete(self) -> None:
        main = set(pd.read_csv(self.corrected / "target_registry.csv")["target_id"])
        controls = {"DFTB_ref-BoE", "Delta_DFTB-ref-BoE"}
        self.assertEqual(len(main), 13)
        self.assertTrue(main.isdisjoint(controls))
        for filename, status_column in (
            ("tmqm_final_ledger.csv", "test_status"),
            ("qm9_final_ledger.csv", "qm9_score_status"),
        ):
            frame = pd.read_csv(self.corrected / filename)
            self.assertEqual(len(frame), 82)
            self.assertTrue(frame["run_name"].is_unique)
            self.assertTrue(frame[status_column].isin({"complete"}).all())
            for _, group in frame.groupby("model"):
                self.assertEqual(set(group["target_id"]), main | controls)
                for target, runs in group.groupby("target_id"):
                    expected = {101} if target in controls else {101, 102, 103}
                    self.assertEqual(set(runs["seed"]), expected)

    def test_prediction_scores_were_reproduced(self) -> None:
        audit = pd.read_csv(self.corrected / "prediction_score_audit.csv")
        self.assertEqual(len(audit), 164)
        self.assertLess(float(audit["absolute_difference"].max()), 1e-12)

    def test_roughness_uses_test_cohorts(self) -> None:
        frame = pd.read_csv(self.corrected / "roughness.csv")
        self.assertEqual(len(frame), 32)
        self.assertTrue(frame["split"].eq("test").all())
        for architecture, rows in frame.groupby("architecture"):
            self.assertIn(architecture, {"GemNet", "SchNet"})
            primary = rows.loc[rows["cohort"].eq("primary")]
            dftb = rows.loc[rows["cohort"].eq("dftb_valid")]
            self.assertEqual(len(primary), 13)
            self.assertTrue(primary["n_molecules"].eq(7104).all())
            self.assertEqual(len(dftb), 3)
            self.assertTrue(dftb["n_molecules"].eq(7068).all())

    def test_dftb_is_excluded_from_pooled_statistics(self) -> None:
        correlations = pd.read_csv(self.corrected / "correlations.csv")
        self.assertFalse(correlations["dftb_included"].astype(bool).any())
        self.assertTrue(correlations["n_targets"].le(13).all())

    def test_validation_snapshots_are_not_active_release_inputs(self) -> None:
        stale = [
            ROOT / "manuscript/si/data/corrected_2026-09-22",
            ROOT / "manuscript/si/figures/corrected_2026-09-22",
            ROOT / "manuscript/si/tables/corrected_2026-09-22",
        ]
        self.assertFalse(any(path.exists() for path in stale))

    def test_manuscript_entry_points_are_present(self) -> None:
        required = [
            ROOT / "manuscript/main.tex",
            ROOT / "manuscript/build/main.pdf",
            ROOT / "manuscript/si/supplementary_information.tex",
            ROOT / "manuscript/si/build/supplementary_information.pdf",
            ROOT / "manuscript/REVISION_STATUS.md",
            ROOT / "manuscript/si/README_CORRECTED_SI.md",
        ]
        self.assertTrue(all(path.is_file() for path in required))


if __name__ == "__main__":
    unittest.main()
