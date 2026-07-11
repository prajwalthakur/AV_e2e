#!/bin/bash
# Downloads Argoverse 2 motion-forecasting train/val/test + top-level
# annotation files. Safe to re-run: uses `s5cmd sync`, so already-downloaded
# files are skipped and only missing/incomplete ones are (re-)fetched.
set -e

export DATASET_NAME=motion-forecasting
export TARGET_DIR=/workspace/workspace/data/av2

mkdir -p "$TARGET_DIR"/{train,val,test}

for split in train val test; do
    echo "==> syncing $split"
    s5cmd --no-sign-request sync \
        "s3://argoverse/datasets/av2/$DATASET_NAME/$split/*" \
        "$TARGET_DIR/$split/"
done

echo "==> syncing top-level annotation files"
s5cmd --no-sign-request cp -n \
    "s3://argoverse/datasets/av2/$DATASET_NAME/av2_mf_focal_test_annotations.parquet" \
    "s3://argoverse/datasets/av2/$DATASET_NAME/av2_mf_multi_test_annotations.parquet" \
    "$TARGET_DIR/"

echo "==> done"
