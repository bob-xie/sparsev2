#!/bin/bash

export HYDRA_FULL_ERROR=1

source "$(dirname "$0")/../cache/path_export.sh"

if [ $# -lt 2 ]; then
    echo "使用方式: $0 <token> <ckpt> [output]"
    echo "示例: $0 6774548111cb5ba4 exp/sparsedrive_agent/2026.06.17.17.46.52/periodic_pdm_ckpts/ep0010.ckpt"
    exit 1
fi

TOKEN=$1
CKPT=$2
OUTPUT=${3:-exp/visualization/trajectory}

cd $NAVSIM_DEVKIT_ROOT

source /home/xqb/DATA2/E2E_Project/miniconda3/bin/activate navsim

python scripts/visualization/visualize_trajectory.py \
    --token $TOKEN \
    --ckpt $CKPT \
    --output $OUTPUT
