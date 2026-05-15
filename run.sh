#!/bin/bash
# Run training locally:   bash run.sh
# Submit to condor GPUs:  condor_submit submit

# Python environment
# I am not sure this will work for this training, maybe one of th elcg envs will work... not sure
PYTHON="/home/hep/mm1221/micromamba/envs/pygnn/bin/python"

# Paths 
SCRIPT_DIR="/vols/cms/mm1221/ReGraphCL/TracksterLinking/TracksterCL"

cd "$SCRIPT_DIR"

$PYTHON train.py \
    --model           static_edge     \
    --loss            negative_mining \
    --hidden_dim      128             \
    --num_layers      3               \
    --contrastive_dim 128             \
    --dropout         0.3             \
    --lr              5e-4            \
    --epochs          220             \
    --batch_size      64              \
    --k_value         24              \
    --patience        300             \
    --train_path      /vols/cms/mm1221/Data/mix/train/ \
    --val_path        /vols/cms/mm1221/Data/mix/val/   \
    --max_train_events 40000          \
    --max_val_events   15000          \
    --output_dir      /vols/cms/mm1221/hgcal/Mixed/Track/TracksterCL/runs/
