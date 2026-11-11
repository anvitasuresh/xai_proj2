# Shortcuts. Each pipeline stage is `bash scripts/run_stage.sh <stage>`; see README.
PY ?= $(shell test -x .venv-data/bin/python && echo .venv-data/bin/python || echo python)

.PHONY: setup test lint label-app calibrate

setup:
	bash scripts/setup_local.sh

test:
	$(PY) -m pytest -q

lint:
	ruff check .

label-app:
	$(PY) -m streamlit run src/tutorbias/labeling/app.py -- --labeler anvita

calibrate:
	$(PY) -m tutorbias.labeling.calibrate --config configs/calibration.yaml --split dev
