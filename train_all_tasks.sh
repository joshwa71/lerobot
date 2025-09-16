#!/bin/bash

# Iterate through tasks 0 to 10 (inclusive)
for task_id in {0..10}; do
    echo "Starting training for task ${task_id}"
    
    lerobot-train \
        --dataset.repo_id="outputs/libero_10_task_${task_id}" \
        --policy.type=act \
        --steps=100000 \
        --batch_size=64 \
        --output_dir="outputs/train/act_libero_10_task_${task_id}" \
        --job_name="act_libero_10_task_${task_id}" \
        --policy.device=cuda \
        --wandb.enable=true \
        --policy.repo_id="outputs/train/act_libero_10_task_${task_id}" \
        --policy.push_to_hub=false
    
    echo "Completed training for task ${task_id}"
done

echo "All training tasks completed"