
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-14B-Instruct
MAX_TOKENS=1024
TEMPERATURE=0.0
PROMPT=plan_and_solve

python eval_longbench_qa.py \
    --stage inference \
    --prompt $PROMPT \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tensor_parallel_size 8 \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --output_path ./results/longbench_qa/$MODEL_NAME_OR_PATH/$PROMPT/predictions

python eval_longbench_qa.py \
    --stage evaluation \
    --eval_strategy sub_em \
    --dataset ./results/longbench_qa/$MODEL_NAME_OR_PATH/$PROMPT/predictions \
    --output_path ./results/longbench_qa/$MODEL_NAME_OR_PATH/$PROMPT/metrics_sub_em.json