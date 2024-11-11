
MAX_MODEL_LEN=65536
GPUT_MEMORY_UTILIZATION=0.90
MAX_TOKENS=1024
TEMPERATURE=0.7

N=8
N_ITER=8
PROMPT=plan_and_solve
ENCODER_MODEL_NAME_OR_PATH=jinaai/jina-embeddings-v3
SCORE=minimum_bayes_risk_score_sentence_embedding

RAW_DATASET=musique_processed_42_4k_31k_0.5k
MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct
DIR_PATH=./results/$RAW_DATASET/${PROMPT}_${TEMPERATURE}_${N}/$MODEL_NAME_OR_PATH

python3 synthesize.py \
    --prompt $PROMPT \
    --raw_dataset ./data/musique/$RAW_DATASET \
    --sample_dataset $DIR_PATH/predictions \
    --sample_dataset_output $DIR_PATH/predictions \
    --score_dataset_output_path $DIR_PATH/$SCORE/predictions \
    --output_path $DIR_PATH/$SCORE/results.json \
    --dataset_output_path $DIR_PATH/$SCORE/dataset \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tensor_parallel_size 8 \
    --max_model_len $MAX_MODEL_LEN \
    --gpu_memory_utilization $GPUT_MEMORY_UTILIZATION \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --n $N \
    --n_iter $N_ITER \
    --encoder_model_name_or_path $ENCODER_MODEL_NAME_OR_PATH 