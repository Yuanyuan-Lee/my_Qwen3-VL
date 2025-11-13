"""
位置token的定义和工具函数
"""
import numpy as np

# 位置token配置
NUM_POSITION_BINS = 1000
POSITION_TOKEN_PREFIX = "<pos_"
POSITION_TOKEN_SUFFIX = ">"

# 坐标范围定义
POSITION_RANGES = {
    'x1': (-0.5, 0.0),
    'x2': (0.0, 0.5),
    'y': (0.0, 0.5),
    'z': (0.0, 1.0)
}

def generate_position_tokens():
    """生成所有位置token: <pos_000> 到 <pos_999>"""
    return [f"{POSITION_TOKEN_PREFIX}{i:03d}{POSITION_TOKEN_SUFFIX}" for i in range(NUM_POSITION_BINS)]

def normalize_position(pos, coord_type):
    """
    归一化位置到[0, 1]
    Args:
        pos: 位置值
        coord_type: 'x1', 'x2', 'y', 'z'
    """
    if coord_type in ['y1', 'y2']:
        coord_type = 'y'
    elif coord_type in ['z1', 'z2']:
        coord_type = 'z'
    
    min_val, max_val = POSITION_RANGES[coord_type]
    return np.clip((pos - min_val) / (max_val - min_val), 0, 1)

def denormalize_position(norm_pos, coord_type):
    """
    反归一化位置
    Args:
        norm_pos: 归一化值 [0, 1]
        coord_type: 'x1', 'x2', 'y', 'z'
    """
    if coord_type in ['y1', 'y2']:
        coord_type = 'y'
    elif coord_type in ['z1', 'z2']:
        coord_type = 'z'
    
    min_val, max_val = POSITION_RANGES[coord_type]
    return norm_pos * (max_val - min_val) + min_val

def position_to_token(pos_value):
    """将归一化位置值[0,1]转换为token id [0, 999]"""
    return int(np.clip(np.round(pos_value * (NUM_POSITION_BINS - 1)), 0, NUM_POSITION_BINS - 1))

def token_to_position(token_id):
    """将token id [0, 999]转换为归一化位置值[0,1]"""
    return token_id / (NUM_POSITION_BINS - 1)

def parse_position_string(position_str):
    """
    解析位置token字符串
    Args:
        position_str: "<pos_123><pos_456><pos_789><pos_012><pos_345><pos_678>"
    Returns:
        list of 6 normalized position values [0, 1]
    """
    import re
    pattern = r"<pos_(\d{3})>"
    matches = re.findall(pattern, position_str)
    if len(matches) != 6:
        raise ValueError(f"Expected 6 position tokens, got {len(matches)}")
    
    token_ids = [int(m) for m in matches]
    positions = [token_to_position(tid) for tid in token_ids]
    return positions

def format_position_tokens(positions):
    """
    将6个归一化位置值格式化为token字符串
    Args:
        positions: list/array of 6 normalized values [0, 1]
    Returns:
        "<pos_xxx><pos_yyy><pos_zzz><pos_xxx><pos_yyy><pos_zzz>"
    """
    if len(positions) != 6:
        raise ValueError(f"Expected 6 positions, got {len(positions)}")
    
    tokens = [f"{POSITION_TOKEN_PREFIX}{position_to_token(p):03d}{POSITION_TOKEN_SUFFIX}" 
              for p in positions]
    return "".join(tokens)