python3 fenge.py \
    /share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_blocks_ranking_8/qwen3_vl_data_blocks_ranking_8.json \
    --train qwen3_vl_data_blocks_ranking_8/train.json \
    --val qwen3_vl_data_blocks_ranking_8/val.json \
    --ratio 0.9 \
    --seed 123

echo "Done fenge qwen3_vl_data_blocks_ranking_8"

sleep 30

python3 fenge.py \
    /share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_handover_8/qwen3_vl_data_handover_8.json \
    --train qwen3_vl_data_handover_8/train.json \
    --val qwen3_vl_data_handover_8/val.json \
    --ratio 0.9 \
    --seed 123

echo "Done fenge qwen3_vl_data_handover_8"
