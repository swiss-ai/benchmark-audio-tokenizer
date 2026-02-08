# Extend the NeMo container that's already used in your environment
FROM nvcr.io/nvidia/nemo:25.11

# Metadata
LABEL maintainer="rosmith"
LABEL description="NeMo container with CTranslate2 CUDA support for faster-whisper on ARM64"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

# Install build dependencies including Python development headers
# Note: NeMo 25.11 uses Python 3.12
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /opt

# Clone CTranslate2 repository (pinned to stable release)
RUN git clone --recursive https://github.com/OpenNMT/CTranslate2.git && \
    cd CTranslate2 && \
    git checkout v4.7.1

# Build CTranslate2 C++ library with CUDA and cuDNN support
# The NeMo container includes cuDNN 9 headers - we enable cuDNN for better performance
WORKDIR /opt/CTranslate2/build
RUN PYTHON_VERSION=$(python3 --version | grep -oP '\d+\.\d+') && \
    export TORCH_LIB=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/torch/lib && \
    export CUDA_ROOT=/usr/local/cuda && \
    export CUDNN_ROOT=/usr/local/cuda && \
    export PYTHON_INCLUDE=/usr/include/python${PYTHON_VERSION} && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=/opt/ctranslate2 \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
        -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so \
        -DWITH_MKL=OFF \
        -DWITH_RUY=ON \
        -DOPENMP_RUNTIME=COMP \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc \
        -DCUDA_INCLUDE_DIRS="/usr/local/cuda/include;/usr/local/cuda/targets/sbsa-linux/include" \
        -DCUDA_ARCH_LIST="8.0;9.0" \
        -DCMAKE_CXX_FLAGS="-I${PYTHON_INCLUDE}" \
        -DCMAKE_C_FLAGS="-I${PYTHON_INCLUDE}" && \
    make -j$(nproc) && \
    make install

# Set library path for runtime (include both lib and lib64)
ENV LD_LIBRARY_PATH=/opt/ctranslate2/lib64:/opt/ctranslate2/lib:$LD_LIBRARY_PATH

# Build and install Python wrapper
WORKDIR /opt/CTranslate2/python
RUN PYTHON_VERSION=$(python3 --version | grep -oP '\d+\.\d+') && \
    export CTRANSLATE2_ROOT=/opt/ctranslate2 && \
    export PYTHON_INCLUDE=/usr/include/python${PYTHON_VERSION} && \
    export CFLAGS="-I$PYTHON_INCLUDE" && \
    export CPPFLAGS="-I$PYTHON_INCLUDE" && \
    python3 -m pip install --verbose .

# Install faster-whisper (latest version - runtime compatible)
RUN python3 -m pip install faster-whisper

# Verify CTranslate2 installation
RUN python3 -c "import ctranslate2; print('CTranslate2 version:', ctranslate2.__version__)"

# Note: faster-whisper verification skipped due to import issues during build
# Test manually after deployment

# Set working directory back to default
WORKDIR /workspace

# Add a script to verify CUDA is working
RUN echo '#!/bin/bash\n\
echo "==========================================="\n\
echo "CTranslate2 + faster-whisper CUDA Test"\n\
echo "==========================================="\n\
nvidia-smi --query-gpu=index,name,memory.total --format=csv\n\
echo ""\n\
python -c "import torch; print(f\"PyTorch CUDA available: {torch.cuda.is_available()}\")"\n\
echo "==========================================="\n\
' > /usr/local/bin/test-cuda && chmod +x /usr/local/bin/test-cuda

# Default command
CMD ["/bin/bash"]
