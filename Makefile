SHELL := /bin/bash
ROOT  := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DATE  := $(shell date '+%Y%m%d-%H%M%S')

export PYTHONDONTWRITEBYTECODE = 1
export PYTHONUNBUFFERED        = 1
export PYTHONPATH              = $(ROOT)/src
export PYTHONOPTIMIZE          = 0
export OMP_NUM_THREADS         = 1
export CUDA_VISIBLE_DEVICES    = 0

export PIP_DEFAULT_TIMEOUT     = 300
export POETRY_REQUESTS_TIMEOUT = 300

export HF_HUB_ETAG_TIMEOUT     = 300
export HF_DATASETS_OFFLINE     = 0

RSYNC               = rsync --archive --verbose --compress --rsh='ssh -o ClearAllForwardings=yes'
REMOTE_HOST         = pp-yolocam
REMOTE_PATH         = projects/yolocam

CONDA_ENV_NAME      = yolocam

# -----------------------------------------------------------------------------
# default
# -----------------------------------------------------------------------------

.DEFAULT_GOAL = detect

# -----------------------------------------------------------------------------
# conda and linux install and configuration for a new machine
# -----------------------------------------------------------------------------

.PHONY: conda-install
conda-install:
	@wget -qc -O '${HOME}/miniconda.sh' 'https://repo.anaconda.com/miniconda/Miniconda3-py312_25.9.1-3-Linux-x86_64.sh'
	@mkdir -p "${HOME}/opt"
	@bash '${HOME}/miniconda.sh' -b -f -p "${HOME}/opt/miniconda"
	@mkdir -p "${HOME}/.local/bin"
	@ln -sfT "${HOME}/opt/miniconda/bin/conda" "${HOME}/.local/bin/conda"
	@rm -vf '${HOME}/miniconda.sh'

.PHONY: conda-setup
conda-setup:
	@conda config --system --set solver libmamba
	@conda tos accept --override-channels --channel 'https://repo.anaconda.com/pkgs/main'
	@conda tos accept --override-channels --channel 'https://repo.anaconda.com/pkgs/r'
	@conda config --system --remove channels defaults
	@conda config --system --add channels conda-forge
	@conda config --system --add channels nvidia
	@conda config --show-sources
	@conda config --show channels

# -----------------------------------------------------------------------------
# conda environment
# -----------------------------------------------------------------------------

.PHONY: env-create
env-create:
	@conda create --yes --copy --name "$(CONDA_ENV_NAME)" \
		conda-forge::python=3.12.12 \
		conda-forge::poetry=2.2.1

.PHONY: env-remove
env-remove:
	@conda env remove --yes --name "$(CONDA_ENV_NAME)"

.PHONY: env-poetry-install
env-poetry-install:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry install --no-root --no-directory

.PHONY: env-poetry-update
env-poetry-update:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry update

.PHONY: env-poetry-list
env-poetry-list:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		poetry show --tree

.PHONY: env-shell
env-shell:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		bash

.PHONY: env-python
env-python:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		python3

.PHONY: env-info
env-info:
	@conda run --no-capture-output --live-stream --name "$(CONDA_ENV_NAME)" \
		conda info

# -----------------------------------------------------------------------------
# linters
# -----------------------------------------------------------------------------

.PHONY: shellcheck
shellcheck:
	@bin/run shellcheck --norc --shell=bash bin/*

.PHONY: lint-flake8
lint-flake8:
	@bin/run flake8 src

.PHONY: lint
lint: shellcheck lint-flake8

# -----------------------------------------------------------------------------
# build
# -----------------------------------------------------------------------------

.PHONY: build
build: lint test

# -----------------------------------------------------------------------------
# run
# -----------------------------------------------------------------------------

.PHONY: cameras
cameras:
	@bin/run python3 src/app.py cameras

.PHONY: detect
detect:
	@bin/run python3 src/app.py detect --camera 4

.PHONY: depth
depth:
	@bin/run python3 src/app.py depth --camera 4

.PHONY: segmentation
segmentation:
	@bin/run python3 src/app.py segmentation --camera 4

# -----------------------------------------------------------------------------
# system
# -----------------------------------------------------------------------------

.PHONY: vmstat
vmstat:
	@vmstat --unit M --timestamp --wide 3 | tee "$(ROOT)/work/vmstat-$(DATE).log"

.PHONY: gpustat
gpustat:
	@nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv --loop=5

# -----------------------------------------------------------------------------
# cleanup
# -----------------------------------------------------------------------------

.PHONY: clean-pycache
clean-pycache:
	@find "$(ROOT)/src" -type d -name '__pycache__' -print0 | xargs -0 -r -n 1 rm --recursive --verbose
	@find "$(ROOT)/tst" -type d -name '__pycache__' -print0 | xargs -0 -r -n 1 rm --recursive --verbose

.PHONY: clean-work-logs
clean-work-logs:
	@find "$(ROOT)/work" -type f -name '*.log' -print0 | xargs -0 -r -n 1 rm --recursive --verbose

.PHONY: clean-work
clean-work: clean-work-logs

.PHONY: clean
clean: clean-pycache clean-work

# -----------------------------------------------------------------------------
# rsync push
# -----------------------------------------------------------------------------

.PHONY: rsync-push
rsync-push:
	@$(RSYNC) \
		--exclude='/.git' \
		--exclude='/.idea' \
		--exclude='/.benchmark' \
		--exclude='*.log' \
		--exclude='__pycache__' \
		--exclude='.pytest_cache' \
		--exclude='.ipynb_checkpoints' \
		'$(ROOT)/' \
		'$(REMOTE_HOST):$(REMOTE_PATH)'

# -----------------------------------------------------------------------------
# rsync pull
# -----------------------------------------------------------------------------

.PHONY: rsync-pull
rsync-pull:
	@$(RSYNC) \
		--exclude='/.git' \
		--exclude='/.idea' \
		--exclude='/.benchmark' \
		--exclude='__pycache__' \
		--exclude='.pytest_cache' \
		--exclude='.ipynb_checkpoints' \
		'$(REMOTE_HOST):$(REMOTE_PATH)' \
		'$(ROOT)/'
