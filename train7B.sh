args='      --model_name_or_path "/mnt/nas-data-5/zengshuang.zs/amap_app_common_h20_nm125/7B_v9.14_scalevln_dagger_r2r_rxr/" \
            --tune_mm_llm True \
            --tune_mm_vision False \
            --tune_mm_mlp True \
            --dataset_use "real_world" \
            --output_dir "/mnt/nas-data-5/zengshuang.zs/amap_app_common_h20_nm125/7B_v9.24_real_world_v1" \
            --cache_dir "./cache" \
            --bf16 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 2 \
            --learning_rate 1e-5 \
            --mm_projector_lr 5e-6 \
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
            --num_train_epochs 8 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 10 \
            --save_strategy "steps" \
            --save_steps 100 \
            --save_total_limit 1 \
            --deepspeed "scripts/zero0.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 64 \
            --group_by_modality_length true \
            --seed 42 \
            --report_to "none" \
            --use_vggt_feature true \
            --vggt_model_path "/mnt/nas-data-5/zengshuang.zs/model/VGGT-1B"\
            --reference_frame first \
            > /mnt/nas-data-5/zengshuang.zs/amap_app_common_h20_nm125/7B_v9.24_real_world_v1/train.log 2>&1
            '

echo "${args}"




nebulactl run mdl --queue=amap_app_common_h20_na175 \
                  --entry=src/qwen_vl/train/train_qwen.py \
                  --algo_name=pytorch240 \
                  --worker_count=64 \
                  --user_params="$args" \
                  --file.cluster_file=./cluster.json \
                  --job_name='JanusVLN' \
                  --launcher=accelerate \
                  --nas_file_system_id=1fff449945-wau24.cn-beijing.nas.aliyuncs.com \
                  --nas_file_system_mount_path=/mnt/nas-data-5 \
                  --public_pool_job_type=queuing \







# args='      --model_name_or_path "/mnt/nas-data-5/zengshuang.zs/model/Qwen2.5-VL-7B-Instruct/" \
#             --tune_mm_llm True \
#             --tune_mm_vision False \
#             --tune_mm_mlp False \
#             --dataset_use "train_scalevln_r2r_rxr" \
#             --output_dir "/mnt/nas-data-5/zengshuang.zs/amap_common_308x_na130/7B_v8.15_scalevln_r2r_rxr_novggt" \
#             --cache_dir "./cache" \
#             --bf16 \
#             --per_device_train_batch_size 1 \
#             --gradient_accumulation_steps 8 \
#             --learning_rate 5e-6 \
#             --mm_projector_lr 1e-5 \
#             --vision_tower_lr 1e-6 \
#             --optim adamw_torch \
#             --model_max_length 16384 \
#             --data_flatten False \
#             --max_pixels $((576*28*28)) \
#             --min_pixels $((16*28*28)) \
#             --base_interval 2 \
#             --video_max_frames 8 \
#             --video_min_frames 4 \
#             --video_max_frame_pixels $((1664*28*28)) \
#             --video_min_frame_pixels $((256*28*28)) \
#             --num_train_epochs 1 \
#             --warmup_ratio 0.03 \
#             --lr_scheduler_type "cosine" \
#             --weight_decay 0.01 \
#             --logging_steps 10 \
#             --save_strategy "steps" \
#             --save_steps 100 \
#             --save_total_limit 1 \
#             --deepspeed "scripts/zero0.json" \
#             --gradient_checkpointing \
#             --dataloader_num_workers 16 \
#             --group_by_modality_length true \
#             --seed 42 \
#             --report_to "none" \
#             --use_vggt_feature false \
#             --vggt_model_path "/mnt/nas-data-5/zengshuang.zs/model/VGGT-1B"\
#             --reference_frame first \
#             > /mnt/nas-data-5/zengshuang.zs/amap_common_308x_na130/7B_v8.15_scalevln_r2r_rxr_novggt/train.log 2>&1
#             '

# echo "${args}"




# nebulactl run mdl --queue=amap_app_common_h20_na175 \
#                   --entry=src/qwen_vl/train/train_qwen.py \
#                   --algo_name=pytorch240 \
#                   --worker_count=96 \
#                   --user_params="$args" \
#                   --file.cluster_file=./cluster.json \
#                   --job_name='vgllm_7B' \
#                   --launcher=accelerate \
#                   --nas_file_system_id=1fff449945-wau24.cn-beijing.nas.aliyuncs.com \
#                   --nas_file_system_mount_path=/mnt/nas-data-5 \
#                   --public_pool_job_type=queuing \