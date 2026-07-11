#!/bin/bash
source "$(dirname "$0")/base.sh"

stop_docker

mode="gpu"
while getopts 'cgh' opt; do
    case "$opt" in
        c) mode="cpu" ;;
        g) mode="gpu" ;;
        ?|h) echo "Usage: $(basename "$0") [-c|-g]"; exit 1 ;;
    esac
done
shift "$(($OPTIND -1))"

: "${PROJECT_ROOT:=$(pwd)}"

if [ "$mode" = "gpu" ]; then
  run_docker --gpus all \
    --mount type=bind,src="${PROJECT_ROOT}",target="/workspace" \
    -- bash
else
  run_docker \
    --mount type=bind,src="${PROJECT_ROOT}",target="/workspace" \
    -- bash
fi