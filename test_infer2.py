from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import time

# default: Load the model on the available device(s)
# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct", dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = AutoModelForImageTextToText.from_pretrained(
    "/share/project/liyuanyuan/code/Qwen3-VL/weights/Qwen3-VL-2B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("/share/project/liyuanyuan/code/Qwen3-VL/weights/Qwen3-VL-2B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                # "max_pixels": 12544,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# ----------------- 新增：推理时间统计 -----------------
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# 可配置：先做若干次 warmup（不计时），然后多次测量取平均
warmup_runs = 1
measure_runs = 5
max_new_tokens = 128

# Warmup
with torch.inference_mode():
    for _ in range(warmup_runs):
        _sync()
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
        _sync()

# 测量多次并统计
run_times = []
generated_lens = []
for i in range(measure_runs):
    _sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    _sync()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    run_times.append(elapsed)

    # 计算本次生成的 token 数（每个样本）
    # inputs.input_ids: [batch, seq_len]; generated_ids: [batch, seq_len_out]
    gen_lens = [out_ids[len(in_ids):].shape[0] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    generated_lens.append(gen_lens)

# 汇总并打印结果
import statistics
avg_time = statistics.mean(run_times)
p50 = statistics.median(run_times)
p95 = sorted(run_times)[int(0.95 * len(run_times)) - 1] if measure_runs > 0 else avg_time

batch_size = inputs.input_ids.size(0)
total_generated_tokens = sum(sum(l) for l in generated_lens)
throughput_tokens_per_sec = total_generated_tokens / sum(run_times) if sum(run_times) > 0 else 0.0

print(f"Inference runs: warmup={warmup_runs}, measured={measure_runs}")
print(f"Batch size: {batch_size}, max_new_tokens: {max_new_tokens}")
print(f"Times (s) per run: {[round(x, 4) for x in run_times]}")
print(f"avg: {avg_time:.4f}s, p50: {p50:.4f}s, p95: {p95:.4f}s")
print(f"Total generated tokens (all runs): {total_generated_tokens}")
print(f"Throughput: {throughput_tokens_per_sec:.2f} tokens/s (across measured runs)")

# 最后再一次实际解码并打印文本（可选）
# 使用最后一次生成的 generated_ids
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("Output text:", output_text)