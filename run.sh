#!/bin/bash
# Run training locally:   bash run.sh
# Submit to condor GPUs:  condor_submit submit

# Python environment
# I am not sure this will work for this training, maybe one of th elcg envs will work... not sure
PYTHON="/home/hep/mm1221/micromamba/envs/pygnn/bin/python"

$PYTHON train.py \
    --model           static_edge     \
    --loss            negative_mining \
    --hidden_dim      128             \
    --num_layers      3               \
    --contrastive_dim 16             \
    --dropout         0.1            \
    --lr              5e-4            \
    --epochs          10             \
    --batch_size      16              \
    --k_value         16             \
    --patience        30             \
    --train_path      /vols/cms/mm1221/Data/100k/5pi/train/ \
    --val_path        /vols/cms/mm1221/Data/100k/5pi/val/   \
    --max_train_events 40          \
    --max_val_events   15         \
    --output_dir      /vols/cms/mm1221/hgcal_gnn/runs/test/
