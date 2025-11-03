python process_data_to_qwen3vl_action.py \
    --root-data-dir /share/project/liyuanyuan/code/RoboTwin/data \
    --save-dir /share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_stack_bowls_4/ \
    --tasks "stack_bowls_two,stack_bowls_three" \
    --levels "clean,randomized" \
    --segment-size 4 \
    --total-segments 16 \
    --total-frames 64