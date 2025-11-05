import json
import os

input_dir = '/share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_stack_bowls_8_action_norm'
output = 'qwen3_vl_data_stack_bowls_8_action_norm/qwen3_vl_data_stack_bowls_8_action_norm.json'

# 获取目录下所有json文件（非递归）
json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]

print(f"找到 {len(json_files)} 个json文件。")

merged = []
for file_path in json_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
        merged += data

with open(output, 'w') as fout:
    json.dump(merged, fout, indent=4)

print(f"已将 {len(json_files)} 个json文件合并为 {output}")