#!/bin/bash

# Download RAVDESS dataset
# Note: You need to download manually from https://zenodo.org/record/1188976
# This script provides instructions and creates the directory structure

echo "======================================================================"
echo "RAVDESS Dataset Download Instructions"
echo "======================================================================"
echo ""
echo "The RAVDESS dataset must be downloaded manually from Zenodo:"
echo "https://zenodo.org/record/1188976"
echo ""
echo "Please follow these steps:"
echo "1. Visit the URL above"
echo "2. Download 'Audio_Speech_Actors_01-24.zip' (1.09 GB)"
echo "3. Extract the ZIP file"
echo "4. Move the extracted 'Actor_*' folders to: data/RAVDESS/"
echo ""
echo "Creating directory structure..."

# Create directories
mkdir -p data/RAVDESS
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p results/plots
mkdir -p results/metrics
mkdir -p results/logs

echo "Directories created:"
echo "  - data/RAVDESS/ (place extracted Actor folders here)"
echo "  - data/processed/"
echo "  - models/checkpoints/"
echo "  - results/plots/"
echo "  - results/metrics/"
echo "  - results/logs/"
echo ""

# Check if dataset exists
if [ -d "data/RAVDESS/Actor_01" ]; then
    echo "✓ Dataset found in data/RAVDESS/"
    
    # Count files
    file_count=$(find data/RAVDESS -name "*.wav" | wc -l)
    actor_count=$(ls -d data/RAVDESS/Actor_* 2>/dev/null | wc -l)
    
    echo "  Files found: $file_count"
    echo "  Actors found: $actor_count"
    
    if [ $file_count -eq 1440 ] && [ $actor_count -eq 24 ]; then
        echo "✓ Dataset is complete!"
    else
        echo "⚠ Dataset may be incomplete. Expected: 1440 files, 24 actors"
    fi
else
    echo "⚠ Dataset not found. Please download and extract to data/RAVDESS/"
fi

echo ""
echo "======================================================================"
echo "Setup complete!"
echo "======================================================================"
