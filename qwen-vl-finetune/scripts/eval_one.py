import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
from PIL import Image
import random

# 配置
model_path = "./qwen3vl_stack_bowls_8_v2_val"
val_file = "/share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_stack_bowls_8_v2/val.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens = 256
output_file = Path("eval_results/qwen3_vl_data_stack_bowls_8_v2_val.txt")

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
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
            if img_idx < len(images):
                content.append({"type": "image", "image": images[img_idx]})
                img_idx += 1
            else:
                content.append({"type": "image", "image": images[-1]})

    return [
        {"role": "user", "content": content},
    ], gt_text

def infer_one(messages):
    prompt = processor.apply_chat_template(messages, tokenize=False)
    images = []
    for msg in messages:
        for seg in msg["content"]:
            if seg["type"] == "image":
                images.append(seg["image"])
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("Prompt:", prompt)
    print("Images:", len(images))
    print("Output shape:", output.shape)
    print("Input length:", inputs["input_ids"].shape[1])
    print("Generated ids:", output[0][inputs["input_ids"].shape[1]:])
    print("Decoded:", tokenizer.decode(output[0][inputs["input_ids"].shape[1]:]))
    return response.strip()

def load_image(img_path):
    if img_path is None:
        return Image.new("RGB", (224, 224), (255, 255, 255))
    return Image.open(img_path).convert("RGB")

def main():
    data = load_data(val_file)
    if len(data) > 16:
        data = random.sample(data, 16)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for item in tqdm(data, desc="Evaluating"):
        messages, gt = build_messages(item)
        pred = infer_one(messages)
        results.append((gt, pred))
        line = f"GT: {gt}\nPred: {pred}\n{'='*30}\n"
        print(line, end="")
    total = len(results)
    correct = sum(1 for gt, pred in results if pred == gt)
    non_empty = [(gt, pred) for gt, pred in results if pred.strip() != ""]
    correct_non_empty = sum(1 for gt, pred in non_empty if pred == gt)
    total_non_empty = len(non_empty)
    summary = (
        f"总准确率: {correct}/{total} = {correct/total:.4f}\n"
        f"排除空输出的准确率: {correct_non_empty}/{total_non_empty} = "
        f"{(correct_non_empty/total_non_empty):.4f}\n" if total_non_empty > 0 else "无非空输出\n"
    )
    print(summary, end="")
    with open(output_file, "w", encoding="utf-8") as fout:
        for gt, pred in results:
            fout.write(f"GT: {gt}\nPred: {pred}\n{'='*30}\n")
        fout.write(summary)

if __name__ == "__main__":
    main()