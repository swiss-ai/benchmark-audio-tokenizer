#!/bin/bash
set -euo pipefail
export PATH=/opt/audio-runtime/ffmpeg/bin:/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_PRELOAD LD_LIBRARY_PATH PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/audio-r2-uv
cd /build/packages
sha256sum -c SHA256SUMS
uv pip install --python /opt/venv/bin/python --no-deps --no-index ./lhotse-*.whl
find /opt/audio-runtime/ffmpeg -type d -exec chmod 0755 {} +
find /opt/audio-runtime/ffmpeg -type f -exec chmod a+rX {} +
cd /build
/opt/venv/bin/python recipe/inventory.py > python-after.json
dpkg-query -W -f='${Package}\t${Version}\n' > dpkg-after.tsv
/opt/venv/bin/python recipe/verify.py
cp package-manifest.json base-actual.sha256 package-diff.json /opt/audio-runtime/
cp recipe/*.sh recipe/*.py /opt/audio-runtime/build-recipe/
ffmpeg -version > ffmpeg-version.txt
