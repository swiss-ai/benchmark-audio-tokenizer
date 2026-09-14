#!/bin/bash
set -euo pipefail
export PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_PRELOAD LD_LIBRARY_PATH PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/audio-build-uv
cd /build
/opt/venv/bin/python recipe/inventory.py > python-before.json
dpkg-query -W -f='${Package}\t${Version}\n' > dpkg-before.tsv
cd packages
sha256sum -c SHA256SUMS
uv pip install --python /opt/venv/bin/python --no-deps --no-index ./*.whl
test ! -e /opt/audio-runtime
mkdir -p /opt/audio-runtime
tar -xf ffmpeg-7.1.1-aarch64.tar -C /opt/audio-runtime
find /opt/audio-runtime/ffmpeg -type d -exec chmod 0755 {} +
find /opt/audio-runtime/ffmpeg -type f -exec chmod a+rX {} +
printf '%s\n' /opt/audio-runtime/ffmpeg/lib > /etc/ld.so.conf.d/audio-ffmpeg.conf
ldconfig
export PATH=/opt/audio-runtime/ffmpeg/bin:$PATH
ffmpeg -version > /build/ffmpeg-version.txt
ffprobe -version > /build/ffprobe-version.txt
ldd /opt/audio-runtime/ffmpeg/bin/ffmpeg > /build/ffmpeg-ldd.txt
if grep -q 'not found' /build/ffmpeg-ldd.txt; then exit 1; fi
cd /build
/opt/venv/bin/python recipe/inventory.py > python-after.json
dpkg-query -W -f='${Package}\t${Version}\n' > dpkg-after.tsv
/opt/venv/bin/python recipe/verify.py
cp package-manifest.json base-actual.sha256 package-diff.json /opt/audio-runtime/
mkdir -p /opt/audio-runtime/build-recipe
cp recipe/*.sh recipe/*.py /opt/audio-runtime/build-recipe/
printf '%s\n' 'NeMo 26.08 WavTokenizer conversion, tokenization and materialization runtime. Lhotse and Polars are installed from pinned offline wheels.' > /opt/audio-runtime/README
