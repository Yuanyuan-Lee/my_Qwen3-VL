import argparse
import json
import os
from typing import Any
import random

def split_list(data_list: list, ratio: float, seed: int):
    rng = random.Random(seed)
    items = list(data_list)
    rng.shuffle(items)
    cut = int(len(items) * ratio)
    return items[:cut], items[cut:]

def split_dict(data_dict: dict, ratio: float, seed: int):
    rng = random.Random(seed)
    items = list(data_dict.items())
    rng.shuffle(items)
    cut = int(len(items) * ratio)
    train_items = items[:cut]
    val_items = items[cut:]
    return dict(train_items), dict(val_items)

def main():
    p = argparse.ArgumentParser(description="按比例将 JSON 切分为训练集和验证集 (支持 list 或 dict)")
    p.add_argument("input", help="输入 JSON 文件路径")
    p.add_argument("--train", default="train.json", help="输出训练集文件 (默认: train.json)")
    p.add_argument("--val", default="val.json", help="输出验证集文件 (默认: val.json)")
    p.add_argument("--ratio", type=float, default=0.9, help="训练集比例 (0-1)，默认 0.9")
    p.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"输入文件不存在: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        train, val = split_list(data, args.ratio, args.seed)
    elif isinstance(data, dict):
        train, val = split_dict(data, args.ratio, args.seed)
    else:
        raise SystemExit("不支持的 JSON 顶层类型，必须是 list 或 dict")

    with open(args.train, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(args.val, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"完成：{len(train)} -> {args.train}, {len(val)} -> {args.val}")

if __name__ == "__main__":
    main()