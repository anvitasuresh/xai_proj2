# Local (CPU) targets. GPU stages run on DCC through slurm/*.sbatch.
PY ?= python

.PHONY: test lint data-items data-messages variants-local label-set calibrate label-app

test:
	$(PY) -m pytest -q

lint:
	ruff check .

data-items:
	$(PY) -m tutorbias.data.build items --config configs/data.yaml

data-messages:
	$(PY) -m tutorbias.data.build messages --config configs/data.yaml

variants-local:   # needs the tutorbias-data env (Multi-VALUE)
	$(PY) -m tutorbias.data.build variants --config configs/data.yaml
	$(PY) -m tutorbias.data.build merge --config configs/data.yaml

label-set:
	$(PY) -m tutorbias.labeling.label_set --config configs/calibration.yaml

label-app:
	streamlit run src/tutorbias/labeling/app.py -- --labeler anvita

calibrate:
	$(PY) -m tutorbias.labeling.calibrate --config configs/calibration.yaml --split dev
