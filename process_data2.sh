python process_data_to_qwen3vl_v1.py \
    --root-data-dir /share/project/liyuanyuan/code/RoboTwin/data \
    --save-dir /share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_stack_bowls_16/ \
    --tasks "stack_bowls_two,stack_bowls_three" \
    --levels "clean,randomized" \
    --segment-size 16 \
    --total-segments 4 \
    --total-frames 64

sleep 300

python process_data_to_qwen3vl_v1.py \
    --root-data-dir /share/project/liyuanyuan/code/RoboTwin/data \
    --save-dir /share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_stack_bowls_32/ \
    --tasks "stack_bowls_two,stack_bowls_three" \
    --levels "clean,randomized" \
    --segment-size 32 \
    --total-segments 2 \
    --total-frames 64