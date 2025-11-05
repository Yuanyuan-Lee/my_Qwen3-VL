python3 fenge.py \
    /share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_stack_bowls_8_action_norm/qwen3_vl_data_stack_bowls_8_action_norm.json \
    --train qwen3_vl_data_stack_bowls_8_action_norm/train.json \
    --val qwen3_vl_data_stack_bowls_8_action_norm/val.json \
    --ratio 0.99 \
    --seed 123

echo "Done fenge qwen3_vl_data_stack_bowls_8_action_norm"
