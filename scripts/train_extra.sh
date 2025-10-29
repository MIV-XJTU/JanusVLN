#!/bin/bash

MASTER_ADDR="127.0.0.1"                    
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  

MODEL_PATH="JanusVLN_Base"  
VGGT_MODEL_PATH="facebook/VGGT-1B"

OUTPUT_DIR="./JanusVLN_Extra"                  
CACHE_DIR="./cache"                        
mkdir -p $OUTPUT_DIR

DATASETS="train_r2r_rxr_extra" 
               

export NCCL_NVLS_ENABLE=0
torchrun --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            src/qwen_vl/train/train_qwen.py \
            --model_name_or_path $MODEL_PATH \
            --vggt_model_path $VGGT_MODEL_PATH \
            --tune_mm_llm True \
            --tune_mm_vision False \
            --tune_mm_mlp True \
            --dataset_use $DATASETS \
            --output_dir $OUTPUT_DIR \
            --cache_dir $CACHE_DIR \
            --bf16 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --learning_rate 2e-5 \
            --mm_projector_lr 1e-5 \
            --vision_tower_lr 1e-6 \
            --optim adamw_torch \
            --model_max_length 163840 \
            --data_flatten False \
            --max_pixels $((576*28*28)) \
            --min_pixels $((16*28*28)) \
            --base_interval 2 \
            --video_max_frames 8 \
            --video_min_frames 4 \
            --video_max_frame_pixels $((1664*28*28)) \
            --video_min_frame_pixels $((256*28*28)) \
            --num_train_epochs 1 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 10 \
            --save_steps 1000 \
            --save_total_limit 1 \
            --deepspeed "scripts/zero3.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 8 \
            --group_by_modality_length true \
            --seed 42 \
            --report_to "none" \
            --reference_frame first \
            > ${OUTPUT_DIR}/train.log 2>&1
