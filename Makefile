SHELL := /bin/bash


# create all venvs (executed from NGC-24.11)
venvs: neucodec cosyvoice2 xcodec2 wavtokenizer glm4voice flexicodec



# xcodec2 with cuda
xcodec2:
	mv .venv-xcodec2 .venv-xcodec2-old || true
	rm -rf .venv-xcodec2-old &

	uv venv .venv-xcodec2 --system-site-packages

	uv pip compile requirements-xcodec2-topdeps.txt -o requirements-xcodec2-subdeps.txt
	sed -i '/^torch==/d' requirements-xcodec2-subdeps.txt
	sed -i '/^torchaudio==/d' requirements-xcodec2-subdeps.txt
	
	source .venv-xcodec2/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/ao.git@v0.6.1 && \
	uv pip install --no-deps -r requirements-xcodec2-subdeps.txt && \
	uv pip install --no-deps xcodec2==0.1.5 && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


# neucodec with cuda
neucodec:
	mv .venv-neucodec .venv-neucodec-old || true
	rm -rf .venv-neucodec-old &

	uv venv .venv-neucodec --system-site-packages

	uv pip compile requirements-neucodec-topdeps.txt -o requirements-neucodec-subdeps.txt
	sed -i '/^torch==/d' requirements-neucodec-subdeps.txt
	sed -i '/^torchaudio==/d' requirements-neucodec-subdeps.txt
	
	source .venv-neucodec/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/ao.git@v0.6.1 && \
	uv pip install --no-deps -r requirements-neucodec-subdeps.txt && \
	uv pip install --no-deps neucodec==0.0.4 && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


# glm4voice
glm4voice:
	mv .venv-glm4voice .venv-glm4voice-old || true
	rm -rf .venv-glm4voice-old &

	uv venv .venv-glm4voice --system-site-packages

	uv pip compile requirements-glm4voice-topdeps.txt -o requirements-glm4voice-subdeps.txt
	sed -i '/^torch==/d' requirements-glm4voice-subdeps.txt
	sed -i '/^torchaudio==/d' requirements-glm4voice-subdeps.txt
	
	source .venv-glm4voice/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps -r requirements-glm4voice-subdeps.txt && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


# wavtokenizer
wavtokenizer:
	mv .venv-wavtokenizer .venv-wavtokenizer-old || true
	rm -rf .venv-wavtokenizer-old &

	uv venv .venv-wavtokenizer --system-site-packages

	uv pip compile requirements-wavtokenizer-topdeps.txt -o requirements-wavtokenizer-subdeps.txt
	sed -i '/^torch==/d' requirements-wavtokenizer-subdeps.txt
	sed -i '/^torchaudio==/d' requirements-wavtokenizer-subdeps.txt

	source .venv-wavtokenizer/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps -r requirements-wavtokenizer-subdeps.txt && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


# cosyvoice2
cosyvoice2:
	mv .venv-cosyvoice2 .venv-cosyvoice2-old || true
	rm -rf .venv-cosyvoice2-old &

	uv venv .venv-cosyvoice2 --system-site-packages

	uv pip compile requirements-cosyvoice2-topdeps.txt -o requirements-cosyvoice2-subdeps.txt
	sed -i '/^torch==/d' requirements-cosyvoice2-subdeps.txt

	source .venv-cosyvoice2/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps -r requirements-cosyvoice2-subdeps.txt && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"

# flexicodec
flexicodec:
	mv .venv-flexicodec .venv-flexicodec-old || true
	rm -rf .venv-flexicodec-old &

	uv venv .venv-flexicodec --system-site-packages

	uv pip compile requirements-flexicodec-topdeps.txt -o requirements-flexicodec-subdeps.txt
	sed -i '/^torch==/d' requirements-flexicodec-subdeps.txt
	sed -i '/^torchaudio==/d' requirements-flexicodec-subdeps.txt

	source .venv-flexicodec/bin/activate && \
	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
	uv pip install --no-deps -r requirements-flexicodec-subdeps.txt && \
	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


# # xcodec2 CPU-only torch
# xcodec2:
# 	mv .venv-xcodec2 .venv-xcodec2-old || true
# 	rm -rf .venv-xcodec2-old &

# 	uv venv .venv-xcodec2
	
# 	source .venv-xcodec2/bin/activate && \
# 	uv pip install --no-deps xcodec2==0.1.5 && \
# 	uv pip install -r requirements-xcodec2-freeze.txt && \
# 	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


# #############################################################################


# # xcodec2 with CUDA intent (not working because torchtune leads to AttributeError: module 'torch' has no attribute 'int1')
# xcodec2:
# 	mv .venv-xcodec2 .venv-xcodec2-old || true
# 	rm -rf .venv-xcodec2-old &

# 	uv venv .venv-xcodec2 --system-site-packages

# 	uv pip compile requirements-xcodec2-topdeps.txt -o requirements-xcodec2-subdeps.txt
# 	sed -i '/^torch==/d' requirements-xcodec2-subdeps.txt
	
# 	source .venv-xcodec2/bin/activate && \
# 	uv pip install --no-deps xcodec2==0.1.5 && \
# 	uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.6 && \
# 	uv pip install --no-deps -r requirements-xcodec2-subdeps.txt && \
# 	python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); import torchaudio; print(f'torchaudio Version: {torchaudio.__version__}')"


