#!/bin/bash
# Reuse this campaign's verified rootfs; export a distinct revision-2 image.
set -euo pipefail
OUT=$(realpath "$1")
PREVIOUS=$(dirname "$OUT")
WORK=$(cat "$PREVIOUS/build-work-path.txt")
case "$WORK" in /tmp/audio-image-build-${SLURM_JOB_ID}-*) ;; *) exit 2 ;; esac
NAME=nemo-audio-tokenization-26.08
FINAL="$OUT/$NAME.aarch64.sqsh"
test ! -e "$FINAL"
test ! -e "$FINAL.incomplete"
export ENROOT_RUNTIME_PATH="$WORK/runtime" ENROOT_DATA_PATH="$WORK/data"
export ENROOT_TEMP_PATH="$WORK" ENROOT_CACHE_PATH="$WORK/cache" ENROOT_MAX_PROCESSORS=16
export ENROOT_SQUASH_OPTIONS='-comp zstd -Xcompression-level 1 -noD -exit-on-error -processors 16'
printf '%s.%s\n' "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$OUT/export-step.txt"
ROOTFS="$ENROOT_DATA_PATH/$NAME"
test -d "$ROOTFS/opt/audio-runtime"
mkdir -p "$ROOTFS/users/xyixuan"
enroot start --root --rw --rc "$WORK/build-rc.sh" --mount "$OUT:/build" "$NAME" bash /build/recipe/update_lhotse_inside.sh
for name in passwd group shadow gshadow; do
    if [ -f "$WORK/identity-backup/$name" ]; then cp "$WORK/identity-backup/$name" "$ROOTFS/etc/$name"; fi
done
rmdir "$ROOTFS/users/xyixuan" 2>/dev/null || true
enroot export --output "$FINAL.incomplete" "$NAME"
chmod 0644 "$FINAL.incomplete"
mv "$FINAL.incomplete" "$FINAL"
sha256sum "$FINAL" > "$FINAL.sha256"
printf '%s\n' "$FINAL" > "$OUT/candidate-path.txt"
