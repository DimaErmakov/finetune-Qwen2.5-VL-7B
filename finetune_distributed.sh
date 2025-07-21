#!/bin/bash

# Set environment variables for debugging (optional)
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export ACCELERATE_DEBUG_VERBOSITY="debug"

# Set CUDA devices - adjust based on your available GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Use 4 GPUs for 7B model

# Launch the distributed training
accelerate launch \
    --main_process_port=29919 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    --num_machines=1 \
    --num_processes=4 \
    --use_deepspeed \
    finetune_distributed.py
