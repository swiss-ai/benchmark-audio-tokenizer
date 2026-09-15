#!/bin/bash
# Run on a compute host, outside the container. OUT contains offline inputs.
set -euo pipefail
OUT=$(realpath "$1")
BASE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_image"])' "$OUT/package-manifest.json")
EXPECTED=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_sha256"])' "$OUT/package-manifest.json")
NAME=nemo-audio-tokenization-26.08
WORK=/tmp/audio-image-build-${SLURM_JOB_ID}-${SLURM_STEP_ID}
FINAL="$OUT/$NAME.aarch64.sqsh"
test ! -e "$FINAL"
test ! -e "$FINAL.incomplete"
mkdir -p "$WORK/runtime" "$WORK/data"
export ENROOT_RUNTIME_PATH="$WORK/runtime" ENROOT_DATA_PATH="$WORK/data"
export ENROOT_TEMP_PATH="$WORK" ENROOT_CACHE_PATH="$WORK/cache"
export ENROOT_MAX_PROCESSORS=16
export ENROOT_SQUASH_OPTIONS='-comp zstd -Xcompression-level 1 -noD -exit-on-error -processors 16'
export ENROOT_UNSQUASH_OPTIONS='-processors 16 -no-progress'
printf '%s\n' "$WORK" > "$OUT/build-work-path.txt"
printf '%s.%s\n' "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$OUT/build-step.txt"
sha256sum "$BASE" > "$OUT/base-actual.sha256"
test "$(cut -d' ' -f1 "$OUT/base-actual.sha256")" = "$EXPECTED"
enroot create --name "$NAME" "$BASE"
ROOTFS="$ENROOT_DATA_PATH/$NAME"
mkdir -p "$ROOTFS/users/xyixuan" "$WORK/identity-backup"
for name in passwd group shadow gshadow; do
    if [ -f "$ROOTFS/etc/$name" ]; then cp "$ROOTFS/etc/$name" "$WORK/identity-backup/$name"; fi
done
printf '#!/bin/bash\nexec "$@"\n' > "$WORK/build-rc.sh"
chmod 0755 "$WORK/build-rc.sh"
enroot start --root --rw --rc "$WORK/build-rc.sh" --mount "$OUT:/build" "$NAME" bash /build/recipe/install.sh
for name in passwd group shadow gshadow; do
    if [ -f "$WORK/identity-backup/$name" ]; then cp "$WORK/identity-backup/$name" "$ROOTFS/etc/$name"; fi
done
rmdir "$ROOTFS/users/xyixuan" 2>/dev/null || true
enroot export --output "$FINAL.incomplete" "$NAME"
chmod 0644 "$FINAL.incomplete"
mv "$FINAL.incomplete" "$FINAL"
sha256sum "$FINAL" > "$FINAL.sha256"
printf '%s\n' "$FINAL" > "$OUT/candidate-path.txt"
