.PHONY: install test figures example audit manifest verify

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) scripts/run_tests.py

figures:
	$(PYTHON) scripts/reproduce_main_figures.py

example:
	$(PYTHON) scripts/build_descriptors.py \
		--extxyz examples/tiny_molecules.extxyz \
		--mode pre+a \
		--output-dir outputs/toy_descriptors
	$(PYTHON) scripts/run_cli.py roughness \
		--features examples/toy_features.csv \
		--targets examples/toy_targets.csv \
		--target-columns parent,complementary_residual,rough_residual \
		--k 3 \
		--output examples/toy_roughness.csv
	$(PYTHON) scripts/run_cli.py screen \
		--input examples/toy_roughness.csv \
		--parent parent \
		--output examples/toy_screening.csv

manifest:
	$(PYTHON) scripts/build_manifest.py

audit:
	$(PYTHON) scripts/verify_release.py

verify: test figures example audit manifest
