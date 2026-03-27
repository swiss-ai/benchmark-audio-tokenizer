SHELL := /bin/bash

nemo_venv:
	mv nemo_venv nemo_venv-old || true
	rm -rf nemo_venv-old &

	uv venv nemo_venv --system-site-packages
	
	source nemo_venv/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.9 && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/Alvorecer721/lhotse.git && \
	uv pip install --no-deps --no-build-isolation nemo_toolkit['asr'] 