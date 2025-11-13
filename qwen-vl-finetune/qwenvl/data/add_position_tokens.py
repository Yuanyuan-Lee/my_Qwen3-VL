"""
向Qwen3-VL tokenizer添加位置预测tokens
"""
from transformers import AutoTokenizer
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from position_tokens import generate_position_tokens

def add_position_tokens_to_tokenizer(model_path, save_path=None):
    """
    向tokenizer添加位置token
    Args:
        model_path: 原始模型路径
        save_path: 保存路径(如果为None,则覆盖原路径)
    """
    if save_path is None:
        save_path = model_path
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 生成位置tokens
    position_tokens = generate_position_tokens()
    
    # 添加特殊tokens
    num_added = tokenizer.add_special_tokens({
        'additional_special_tokens': position_tokens
    })
    
    print(f"Added {num_added} position tokens to tokenizer")
    print(f"New vocabulary size: {len(tokenizer)}")
    
    # 保存tokenizer
    tokenizer.save_pretrained(save_path)
    print(f"Saved tokenizer to {save_path}")
    
    return tokenizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save modified tokenizer")
    args = parser.parse_args()
    
    add_position_tokens_to_tokenizer(args.model_path, args.save_path)