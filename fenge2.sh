python3 fenge.py \
    /share/project/liyuanyuan/code/my_Qwen3-VL/data/qwen3_vl_data_stack_bowls_8_v3/qwen3_vl_data_stack_bowls_8_v3.json \
    --train data/qwen3_vl_data_stack_bowls_8_v3/train.json \
    --val data/qwen3_vl_data_stack_bowls_8_v3/val.json \
    --ratio 0.99 \
    --seed 123

echo "Done fenge data/qwen3_vl_data_stack_bowls_8_v3"
