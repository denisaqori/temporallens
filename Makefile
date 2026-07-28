UV ?= uv
PYTHON := .venv/bin/python

.PHONY: setup setup-cloud kernel verify test test-worktree format lint debug train-baseline \
	debug-adapter smoke-1b train-adapter-cloud eval-noise eval-robustness report

ROBUSTNESS_CONFIGS := \
	configs/experiment/foundation/robustness_noise.yaml \
	configs/experiment/foundation/robustness_channel_dropout.yaml \
	configs/experiment/foundation/robustness_amplitude_scaling.yaml

setup:
	$(UV) sync --python 3.11 --extra dev

setup-cloud:
	@echo "Install the host's CUDA PyTorch build first, then run this target."
	$(UV) venv --python 3.11 --allow-existing
	$(UV) pip install --python $(PYTHON) -r requirements-cloud.txt

kernel:
	$(PYTHON) -m ipykernel install --user --name temporallens \
		--display-name "TemporalLens"

verify:
	$(PYTHON) scripts/verify_environment.py

test:
	$(PYTHON) -m pytest tests/

# Integration tests for scripts/worktree.sh. Not in `test`: drives git end to end
# (clones, branches, pushes to a throwaway local remote) and is slower.
test-worktree:
	bash tests/test_worktree.sh

format:
	$(PYTHON) -m black src scripts tests
	$(PYTHON) -m ruff check src scripts tests --fix

lint:
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m mypy src

debug:
	$(PYTHON) scripts/train_encoder.py --config configs/experiment/foundation/debug_tiny.yaml

train-baseline:
	$(PYTHON) scripts/train_encoder.py \
		--config configs/experiment/foundation/baseline_cnn_subject_split.yaml

debug-adapter:
	$(PYTHON) scripts/train_adapter.py \
		--config configs/experiment/language/adapter_mock_debug.yaml

smoke-1b:
	$(PYTHON) scripts/train_adapter.py \
		--config configs/experiment/language/adapter_llama1b_local.yaml

train-adapter-cloud:
	$(PYTHON) -m accelerate.commands.launch scripts/train_adapter.py \
		--config configs/experiment/language/adapter_llama3b_subject_split.yaml

eval-noise:
	$(PYTHON) scripts/evaluate.py --config configs/experiment/foundation/robustness_noise.yaml

eval-robustness:
	@for cfg in $(ROBUSTNESS_CONFIGS); do \
		$(PYTHON) scripts/evaluate.py --config $$cfg || exit $$?; \
	done

report:
	$(PYTHON) scripts/make_report.py --run-dir results/runs/latest
