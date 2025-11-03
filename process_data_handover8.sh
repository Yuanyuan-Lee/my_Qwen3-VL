python process_data_to_qwen3vl_v1.py \
    --root-data-dir /share/project/liyuanyuan/code/RoboTwin/data \
    --save-dir /share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_handover_8/ \
    --tasks "handover_mic,handover_block" \
    --levels "clean,randomized" \
    --segment-size 8 \
    --total-segments 8 \
    --total-frames 64