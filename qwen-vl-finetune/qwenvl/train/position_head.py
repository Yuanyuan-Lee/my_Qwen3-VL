import torch
import torch.nn as nn

class PositionPredictionHead(nn.Module):
    """
    位置预测头：从语言模型的hidden states预测连续的3D位置
    """
    def __init__(self, hidden_size, num_segments=8, position_dim=6):
        super().__init__()
        self.num_segments = num_segments
        self.position_dim = position_dim
        
        # 位置预测网络
        self.position_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, num_segments * position_dim),
            nn.Sigmoid()  # 输出[0, 1]范围的归一化位置
        )
    
    def forward(self, hidden_states, position_token_indices=None):
        """
        hidden_states: (batch_size, seq_len, hidden_size)
        position_token_indices: (batch_size,) 指示position token的位置
        
        返回: (batch_size, num_segments, position_dim)
        """
        batch_size = hidden_states.shape[0]
        
        if position_token_indices is not None:
            # 提取position token对应的hidden state
            pos_hidden = hidden_states[torch.arange(batch_size), position_token_indices]
        else:
            # 使用最后一个token的hidden state
            pos_hidden = hidden_states[:, -1, :]
        
        # 预测位置
        positions_flat = self.position_mlp(pos_hidden)
        
        # reshape: (batch_size, num_segments, position_dim)
        positions = positions_flat.view(batch_size, self.num_segments, self.position_dim)
        
        return positions