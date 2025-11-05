python process_data_to_qwen3vl_action.py \
    --root-data-dir /share/project/liyuanyuan/code/RoboTwin/data \
    --save-dir /share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_blocks_ranking_8_action/ \
    --tasks "blocks_ranking_rgb,blocks_ranking_size" \
    --levels "clean,randomized" \
    --segment-size 8 \
    --total-segments 8 \
    --total-frames 64