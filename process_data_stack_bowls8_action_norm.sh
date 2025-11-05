python process_data_to_qwen3vl_action_norm.py \
    --root-data-dir /share/project/liyuanyuan/code/RoboTwin/data \
    --save-dir /share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_stack_bowls_8_action_norm/ \
    --tasks "stack_bowls_two,stack_bowls_three" \
    --levels "clean" \
    --segment-size 8 \
    --total-segments 8 \
    --total-frames 64 \
    --epsilons 0.016212109,0.064852099,0.073177666,0.039741902,0.009979329,0.019406529,0.02,0.015658896,0.054141098,0.058059113,0.035590937,0.007510029,0.017089986,0.02
