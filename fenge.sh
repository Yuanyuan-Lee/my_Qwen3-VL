python3 fenge.py \
    /share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_stack_bowls_4/qwen3_vl_data_stack_bowls_4.json \
    --train qwen3_vl_data_stack_bowls_4/train.json \
    --val qwen3_vl_data_stack_bowls_4/val.json \
    --ratio 0.9 \
    --seed 123

echo "Done fenge qwen3_vl_data_stack_bowls_4"

sleep 30

python3 fenge.py \
    /share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_stack_bowls_16/qwen3_vl_data_stack_bowls_16.json \
    --train qwen3_vl_data_stack_bowls_16/train.json \
    --val qwen3_vl_data_stack_bowls_16/val.json \
    --ratio 0.9 \
    --seed 123

echo "Done fenge qwen3_vl_data_stack_bowls_16"

sleep 30

python3 fenge.py \
    /share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_stack_bowls_32/qwen3_vl_data_stack_bowls_32.json \
    --train qwen3_vl_data_stack_bowls_32/train.json \
    --val qwen3_vl_data_stack_bowls_32/val.json \
    --ratio 0.9 \
    --seed 123

echo "Done fenge qwen3_vl_data_stack_bowls_32"