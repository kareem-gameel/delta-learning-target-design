# QM9 total electronic energies

This dataset contains one Psi4 total electronic energy per QM9 molecule. The authors confirm that the calculations used ωB97M-V/def2-SVPD, matching tmQM, and that the exported energy is in Hartree. IDs and numerical values were checked against the study files.

Columns: mol_id is the original QM9 identifier; total_electronic_energy_Ha preserves the cleaned export's value as written; total_electronic_energy_kcal_per_mol is that value multiplied by 627.5094740631 and rounded to nine decimal places. These are absolute electronic energies. The kcal/mol column expresses the same energy per mole of identical molecules.

All 130,807 unique IDs match the study transfer-target table. The common cross-baseline transfer evaluation used 128,988 of these molecules. The raw export contained 260,723 records; 129,916 IDs occurred twice and 891 occurred once. Maximum absolute raw-to-clean difference: 7E-14 Ha. Maximum round-trip conversion discrepancy when re-reading the exported CSV: 7.968E-13 Ha.

SHA-256:
- qm9_total_energies_psi4.csv: 908f985007a0ac7a68d5e6f80d9ef7e11b4de96ea5c3514cc343e98325986ff7
- qm9_total_energies_psi4_raw.csv: 4b7f1e3a73af7024206fcb4b2b9574bab1a80959be4cd8c284dd58ff3ac86c95
- qm9_psi4_total_energies.csv: 075567bb6dd2156699d94f0fce1c93e7a3a788fafaebc41dc981673ff97fc779
