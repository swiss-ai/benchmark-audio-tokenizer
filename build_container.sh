#!/bin/bash
#SBATCH --job-name=build_ctranslate2
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=container_build_%j.out

# Change to the directory with Dockerfile
cd /iopsstor/scratch/cscs/$USER/audio-segmentation/benchmark-audio-tokenizer

echo "=========================================="
echo "Building CTranslate2 Container with cuDNN"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Build the container image with podman
echo "Building container image (this will take 30-60 minutes)..."
podman build -t ctranslate2-nemo-cudnn:latest .

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Container build successful!"
    echo "=========================================="
    echo ""

    # Convert to Enroot squashfs format directly from podman
    echo "Converting to Enroot squashfs format..."
    SQSH_FILE="$SCRATCH/ctranslate2-nemo-cudnn.sqsh"

    # Remove old file if exists
    rm -f "$SQSH_FILE"

    enroot import -o "$SQSH_FILE" podman://ctranslate2-nemo-cudnn:latest

    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ Squashfs file created successfully!"
        echo "=========================================="
        echo "Location: $SQSH_FILE"
        echo "Size: $(du -h $SQSH_FILE | cut -f1)"
        echo ""
        echo "End time: $(date)"
        echo ""
        echo "To use this container, update container.toml:"
        echo "  image = \"$SQSH_FILE\""
    else
        echo ""
        echo "⚠️  Enroot import failed"
        echo "Image is available in podman registry: ctranslate2-nemo-cudnn:latest"
        echo "You can manually convert with:"
        echo "  enroot import -o $SQSH_FILE podman://ctranslate2-nemo-cudnn:latest"
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ Build failed!"
    echo "=========================================="
    echo "Check the output above for errors."
    exit 1
fi
