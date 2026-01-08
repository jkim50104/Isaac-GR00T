#!/usr/bin/env bash
set -euo pipefail

BASE_DATA_PATH="/data1/jokim"
DATA_SOURCE_PATH="pearl:/home/jokim/projects/Isaac-GR00T/data/jkim50104"

# Ensure base dirs
mkdir -p "${BASE_DATA_PATH}/projects/Isaac-GR00T/output"
mkdir -p "${BASE_DATA_PATH}/datasets/lerobot"

# Create/refresh symlinks in the current working directory
ln -sfn "${BASE_DATA_PATH}/projects/Isaac-GR00T/output" output
ln -sfn "${BASE_DATA_PATH}/datasets/lerobot" data

# Copy dataset (avoid accidental merges)
DEST="${BASE_DATA_PATH}/datasets/lerobot/$(basename "${DATA_SOURCE_PATH}")"
if [[ -e "${DEST}" ]]; then
  echo "Destination already exists: ${DEST}"
  echo "Remove it or change destination before copying."
  exit 1
fi

scp -r "${DATA_SOURCE_PATH}" "${BASE_DATA_PATH}/datasets/lerobot/"
echo "Done."
