#!/usr/bin/env bash
set -euo pipefail

# Dataset:
# Data underlying the publication: 3D pose of tomato peduncle nodes using
# deep keypoint detection and point cloud
# DOI: 10.4121/bca55ad7-da67-408d-a083-01376124f514.v1

OUT_DIR="${OUT_DIR:-datasets/wur_peduncle_keypoints}"
ZIP_PATH="${OUT_DIR}/keypoints_dataset.zip"
README_PATH="${OUT_DIR}/README.txt"

README_URL="https://data.4tu.nl/file/bca55ad7-da67-408d-a083-01376124f514/4b13d9f0-8611-465b-810d-969147ee918b"
ZIP_URL="https://data.4tu.nl/file/bca55ad7-da67-408d-a083-01376124f514/255169df-1d76-4a1f-8caf-2721f700db0d"

mkdir -p "${OUT_DIR}"

echo "Downloading README to ${README_PATH}"
wget -c -O "${README_PATH}" "${README_URL}"

echo "Downloading dataset zip to ${ZIP_PATH}"
wget -c -O "${ZIP_PATH}" "${ZIP_URL}"

if command -v md5sum >/dev/null 2>&1; then
  echo "Checking MD5 for ${ZIP_PATH}"
  echo "01b5927b65a90e682c8cdf4df57d41fc  ${ZIP_PATH}" | md5sum -c -
fi

if [[ "${UNZIP_DATASET:-1}" == "1" ]]; then
  echo "Unzipping into ${OUT_DIR}"
  unzip -q -o "${ZIP_PATH}" -d "${OUT_DIR}"
fi

echo "Done. Dataset directory: ${OUT_DIR}"

