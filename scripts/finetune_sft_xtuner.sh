
export MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"
export DATASET="./results/musique_processed_42_4k_31k_2k/plan_and_solve_0.7_32/meta-llama/Llama-3.1-8B-Instruct/minimum_bayes_risk_score_sentence_embedding/dataset"
export LR="5e-6"

CONFIG_NAME=xtuner_sft
OUTPUT_DIR=./checkpoints/Llama-3.1-8B-Instruct-SFT-Full-$LR-musique_processed_42_4k_31k_2k-32

NPROC_PER_NODE=8 xtuner train ./configs/$CONFIG_NAME.py --deepspeed deepspeed_zero3

xtuner convert pth_to_hf ./work_dirs/$CONFIG_NAME/$CONFIG_NAME.py ./work_dirs/$CONFIG_NAME/epoch_1.pth $OUTPUT_DIR --max-shard-size 5GB --safe-serialization