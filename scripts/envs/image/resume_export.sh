#!/bin/bash
# Resume only after package verification succeeded and before export started.
set -euo pipefail
OUT=$(realpath "$1")
WORK=$(cat "$OUT/build-work-path.txt")
case "$WORK" in /tmp/audio-image-build-${SLURM_JOB_ID}-*) ;; *) exit 2 ;; esac
NAME=nemo-audio-tokenization-26.08
FINAL="$OUT/$NAME.aarch64.sqsh"
test ! -e "$FINAL"
test ! -e "$FINAL.incomplete"
test -s "$OUT/package-diff.json"
export ENROOT_RUNTIME_PATH="$WORK/runtime" ENROOT_DATA_PATH="$WORK/data"
export ENROOT_TEMP_PATH="$WORK" ENROOT_CACHE_PATH="$WORK/cache" ENROOT_MAX_PROCESSORS=16
export ENROOT_SQUASH_OPTIONS='-comp zstd -Xcompression-level 1 -noD -exit-on-error -processors 16'
ROOTFS="$ENROOT_DATA_PATH/$NAME"
test -d "$ROOTFS/opt/audio-runtime"
printf '%s.%s\n' "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$OUT/export-step.txt"
enroot start --root --rw --rc "$WORK/build-rc.sh" --mount "$OUT:/build" "$NAME" bash -c '
set -euo pipefail
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_PRELOAD LD_LIBRARY_PATH PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH=/opt/audio-runtime/ffmpeg/bin:/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
/opt/venv/bin/python /build/recipe/inventory.py > /build/python-after.json
dpkg-query -W -f="\${Package}\\t\${Version}\\n" > /build/dpkg-after.tsv
/opt/venv/bin/python /build/recipe/verify.py
cp /build/package-manifest.json /build/base-actual.sha256 /build/package-diff.json /opt/audio-runtime/
mkdir -p /opt/audio-runtime/build-recipe
cp /build/recipe/*.sh /build/recipe/*.py /opt/audio-runtime/build-recipe/
printf "%s\n" "NeMo 26.08 WavTokenizer inference runtime. Custom atomic SHAR preparation APIs are not included." > /opt/audio-runtime/README
'
for name in passwd group shadow gshadow; do
    if [ -f "$WORK/identity-backup/$name" ]; then cp "$WORK/identity-backup/$name" "$ROOTFS/etc/$name"; fi
done
rmdir "$ROOTFS/users/xyixuan" 2>/dev/null || true
enroot export --output "$FINAL.incomplete" "$NAME"
chmod 0644 "$FINAL.incomplete"
mv "$FINAL.incomplete" "$FINAL"
sha256sum "$FINAL" > "$FINAL.sha256"
printf '%s\n' "$FINAL" > "$OUT/candidate-path.txt"
