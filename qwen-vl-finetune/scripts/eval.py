import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import Qwen3VLForConditionalGeneration  # 添加这一行
import concurrent.futures
import math
from tqdm import tqdm  # 新增

# 配置
model_path = "./qwen3vl_stack_bowls_4"  # 微调后模型目录
val_file = "/share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_stack_bowls_4/val.json"                 # 验证集路径
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens = 128
output_file = Path("eval_results/qwen3_vl_data_stack_bowls_4.txt")
BATCH_SIZE = 16  # 可根据显存调整

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(  # 修改这里
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device)
# 不要用 DataParallel
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs for inference")
#     model = torch.nn.DataParallel(model)
model.eval()

def load_data(path):
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f]
    else:
        with open(path) as f:
            return json.load(f)

def build_messages(item):
    conversations = item["conversations"]
    user_turn = next(conv for conv in conversations if conv["from"] == "human")
    gt_turn = next(conv for conv in conversations if conv["from"] == "gpt")
    user_text = user_turn["value"]
    gt_text = gt_turn["value"]

    # 多张图片输入处理
    images = item["image"]
    if not isinstance(images, list):
        images = [images]
    img_idx = 0

    content = []
    user_parts = user_text.split("<image>")
    for i, seg in enumerate(user_parts):
        if seg:
            content.append({"type": "text", "text": seg})
        if i < len(user_parts) - 1:
            # 依次插入图片
            if img_idx < len(images):
                content.append({"type": "image", "image": images[img_idx]})
                img_idx += 1
            else:
                # 如果图片数量不够，也可以选择插入最后一张或报错
                content.append({"type": "image", "image": images[-1]})

    return [
        {"role": "user", "content": content},
    ], gt_text

def infer_one(messages):
    # 1. 生成 prompt
    prompt = processor.apply_chat_template(messages, tokenize=False)
    # 2. 构造 processor 输入
    # 提取图片
    images = []
    for msg in messages:
        for seg in msg["content"]:
            if seg["type"] == "image":
                images.append(seg["image"])
    # 3. 用 processor 处理文本和图片
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    # 4. 推理
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

def batch_infer(batch_messages):
    prompts = []
    images_list = []
    for messages in batch_messages:
        prompt = processor.apply_chat_template(messages, tokenize=False)
        prompts.append(prompt)
        # 提取图片
        images = []
        for msg in messages:
            for seg in msg["content"]:
                if seg["type"] == "image":
                    images.append(seg["image"])
        images_list.append(images)
    # 批量处理
    inputs = processor(text=prompts, images=images_list, return_tensors="pt", padding=True)
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    # 解码
    responses = []
    for i in range(len(prompts)):
        input_len = inputs["input_ids"][i].shape[0]
        resp = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
        responses.append(resp.strip())
    return responses

def eval_one(item):
    messages, gt = build_messages(item)
    pred = infer_one(messages)
    return gt, pred

def main():
    data = load_data(val_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results = []
    all_gt = []
    all_messages = []
    for item in data:
        messages, gt = build_messages(item)
        all_messages.append(messages)
        all_gt.append(gt)
    num_batches = math.ceil(len(all_messages) / BATCH_SIZE)
    for i in tqdm(range(num_batches), desc="Evaluating"):  # 加进度条
        batch_messages = all_messages[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_gts = all_gt[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_preds = batch_infer(batch_messages)
        for gt, pred in zip(batch_gts, batch_preds):
            results.append((gt, pred))
            line = f"GT: {gt}\nPred: {pred}\n{'='*30}\n"
            print(line, end="")
    total = len(results)
    correct = sum(1 for gt, pred in results if pred == gt)
    summary = f"准确率: {correct}/{total} = {correct/total:.4f}\n"
    print(summary, end="")
    with open(output_file, "w", encoding="utf-8") as fout:
        for gt, pred in results:
            fout.write(f"GT: {gt}\nPred: {pred}\n{'='*30}\n")
        fout.write(summary)

if __name__ == "__main__":
    main()