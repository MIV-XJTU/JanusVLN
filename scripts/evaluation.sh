export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="JanusVLN_Extra"
echo "CHECKPOINT: ${CHECKPOINT}"
OUTPUT_PATH="evaluation"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
CONFIG="config/vln_r2r.yaml"
echo "CONFIG: ${CONFIG}"

torchrun --nproc_per_node=8 --master_port=$MASTER_PORT src/evaluation.py --model_path $CHECKPOINT --habitat_config_path $CONFIG --save_video --output_path $OUTPUT_PATH

