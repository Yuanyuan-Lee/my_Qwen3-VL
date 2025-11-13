import torch
import torch.nn as nn
import re
import numpy as np

class PositionPredictor(nn.Module):
    """
    位置预测器：从模型输出中提取并计算位置预测损失
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def parse_position_labels(self, text):
        """
        从文本中解析位置标签
        格式: <pos>x1,y1,z1,x2,y2,z2</pos>_<pos>x1,y1,z1,x2,y2,z2</pos>_...
        返回: tensor of shape (num_segments, 6)
        """
        pattern = r'<pos>([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+)</pos>'
        matches = re.findall(pattern, text)
        
        if not matches:
            return None
        
        positions = []
        for match in matches:
            pos = [float(x) for x in match]
            positions.append(pos)
        
        return torch.tensor(positions, dtype=torch.float32)
    
    def compute_loss(self, predicted_positions, target_positions):
        """
        计算位置预测损失
        predicted_positions: (batch_size, num_segments, 6)
        target_positions: (batch_size, num_segments, 6)
        """
        if predicted_positions is None or target_positions is None:
            return None
        
        # 确保值在[0, 1]范围内
        predicted_positions = torch.clamp(predicted_positions, 0.0, 1.0)
        
        # 计算MSE损失
        loss = self.mse_loss(predicted_positions, target_positions)
        
        return loss
    
    def denormalize_positions(self, normalized_pos):
        """
        反归一化位置预测
        normalized_pos: (batch_size, num_segments, 6) or (num_segments, 6)
        返回真实物理坐标
        """
        ranges = torch.tensor([
            [0.5, -0.5],   # x1: [-0.5, 0]
            [0.5, 0.0],    # y1: [0, 0.5]
            [1.0, 0.0],    # z1: [0, 1]
            [0.5, 0.0],    # x2: [0, 0.5]
            [0.5, 0.0],    # y2: [0, 0.5]
            [1.0, 0.0],    # z2: [0, 1]
        ], device=normalized_pos.device)
        
        real_pos = normalized_pos * ranges[:, 0] + ranges[:, 1]
        return real_pos