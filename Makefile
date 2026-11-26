.PHONY: help install data train eval ab serve test lint all doctor clean-check

help:
	@echo "Helmet Safety v2 — targets:"
	@echo "  install   install runtime + dev dependencies"
	@echo "  data      download + prepare the real dataset"
	@echo "  train     train YOLOv8n"
	@echo "  eval      evaluate on the held-out test split"
	@echo "  ab        single-variable A/B experiments"
	@echo "  serve     REST API + dashboard on :8000"
	@echo "  test      pytest"
	@echo "  lint      ruff"
	@echo "  doctor    verify environment and artifacts"

install:
	python -m pip install -r requirements-dev.txt

data:
	python scripts/00_fetch_weights.py
	python scripts/01_download_dataset.py
	python scripts/02_prepare_dataset.py

train:
	python scripts/03_train.py

eval:
	python scripts/04_evaluate.py

ab:
	python scripts/05_run_ab.py

serve:
	python -m helmet_safety.cli serve

test:
	pytest -v

lint:
	ruff check src scripts tests

doctor:
	python -m helmet_safety.cli doctor

all: data train eval ab
