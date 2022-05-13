SHELL := /bin/bash

HPO := step_2_train_hpo_ml.py
REFIT := step_3_train_refit_ml.py


CPU_LIST := '1-30'
CUDA_VISIBLE_DEVICES := '2'


.PHONY: list venv clean prep pred hpo

list: # list packages
	conda list --prefix venv/

install: # install dependencies in venv folder
	conda create --prefix venv python=3.8 pip cudatoolkit=11.2 -yq && conda run --prefix venv/ python -m pip install -r requirements.txt

clean: # uninstall venv folder
	conda uninstall --prefix venv/ -y --all

hpo: # pred
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) taskset --cpu-list $(CPU_LIST) conda run --prefix venv/ python  $(HPO)

refit: # pred
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) taskset --cpu-list $(CPU_LIST) conda run --prefix venv/ python  $(REFIT)
