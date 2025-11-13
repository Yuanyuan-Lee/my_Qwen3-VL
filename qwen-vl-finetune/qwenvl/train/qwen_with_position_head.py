import os
import sys

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    # 如果Qwen3VL不存在，尝试Qwen2VL
    try:
        from transformers import Qwen2VLForConditionalGeneration as Qwen3VLForConditionalGeneration
    except ImportError:
        raise ImportError("Could not find Qwen3VL or Qwen2VL model class")

from qwenvl.train.position_head import PositionPredictionHead

class Qwen3VLWithPositionHead(nn.Module):
    """
    在Qwen3VL基础上添加位置预测头
    """
    def __init__(self, base_model_name_or_path, num_segments=8, position_dim=6, use_position_head=True):
        super().__init__()
        
        # 加载基础模型 - Qwen3VL
        print(f"Loading Qwen3VL model from {base_model_name_or_path}...")
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True  # Qwen3VL可能需要这个参数
        )
        
        self.use_position_head = use_position_head
        
        if self.use_position_head:
            # 获取hidden size - Qwen3VL的配置结构可能不同
            # 尝试多种方式获取hidden_size
            if hasattr(self.base_model.config, 'hidden_size'):
                hidden_size = self.base_model.config.hidden_size
            elif hasattr(self.base_model.config, 'text_config') and hasattr(self.base_model.config.text_config, 'hidden_size'):
                # Qwen3VL可能将hidden_size放在text_config中
                hidden_size = self.base_model.config.text_config.hidden_size
            elif hasattr(self.base_model.config, 'd_model'):
                hidden_size = self.base_model.config.d_model
            elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embed_dim'):
                hidden_size = self.base_model.model.embed_dim
            else:
                # 打印配置信息帮助调试
                print("Config attributes:", dir(self.base_model.config))
                if hasattr(self.base_model.config, 'text_config'):
                    print("Text config attributes:", dir(self.base_model.config.text_config))
                # 默认值 - 根据模型大小设置
                if "2B" in base_model_name_or_path or "2b" in base_model_name_or_path:
                    hidden_size = 1536  # Qwen3VL-2B
                    print(f"Warning: Could not find hidden_size in config, using default {hidden_size} for 2B model")
                elif "7B" in base_model_name_or_path or "7b" in base_model_name_or_path:
                    hidden_size = 3584  # Qwen3VL-7B
                    print(f"Warning: Could not find hidden_size in config, using default {hidden_size} for 7B model")
                else:
                    hidden_size = 1536  # 默认使用2B的大小
                    print(f"Warning: Could not determine model size, using default hidden_size={hidden_size}")
            
            print(f"Using hidden_size: {hidden_size}")
            
            # 添加位置预测头
            self.position_head = PositionPredictionHead(
                hidden_size=hidden_size,
                num_segments=num_segments,
                position_dim=position_dim
            )
            
            # 位置预测token的ID
            self.position_token_id = None
            
            print(f"Position prediction head initialized: {num_segments} segments x {position_dim} dimensions")
        
        self.config = self.base_model.config
        
    def set_position_token_id(self, token_id):
        """设置位置预测token的ID"""
        self.position_token_id = token_id
        print(f"Position token ID set to: {token_id}")
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        **kwargs
    ):
        """
        前向传播，同时输出language model的结果和位置预测
        """
        # 1. 调用Qwen3VL基础模型
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 需要hidden states用于位置预测
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            **kwargs
        )
        
        # 2. 位置预测
        predicted_positions = None
        position_token_indices = None
        
        if self.use_position_head and self.training:
            # 获取最后一层的hidden states
            hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
            
            # 找到position token的位置
            if self.position_token_id is not None and input_ids is not None:
                # 查找每个样本中最后一个position token的位置
                position_token_mask = (input_ids == self.position_token_id)
                
                # 获取每个batch中最后一个position token的索引
                batch_size = input_ids.shape[0]
                position_token_indices = []
                
                for i in range(batch_size):
                    positions = torch.where(position_token_mask[i])[0]
                    if len(positions) > 0:
                        # 使用最后一个position token
                        position_token_indices.append(positions[-1].item())
                    else:
                        # 如果没有找到，使用序列的最后一个非padding token
                        if attention_mask is not None:
                            non_pad_positions = torch.where(attention_mask[i])[0]
                            if len(non_pad_positions) > 0:
                                position_token_indices.append(non_pad_positions[-1].item())
                            else:
                                position_token_indices.append(input_ids.shape[1] - 1)
                        else:
                            position_token_indices.append(input_ids.shape[1] - 1)
                
                position_token_indices = torch.tensor(
                    position_token_indices, 
                    device=input_ids.device
                )
            
            # 使用position head预测位置
            predicted_positions = self.position_head(
                hidden_states, 
                position_token_indices
            )
        
        # 3. 返回结果（保持与Qwen3VL输出格式一致）
        # 创建一个类似ModelOutput的字典
        output_dict = {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'past_key_values': outputs.past_key_values,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
        
        # 添加Qwen3VL特有的输出
        if hasattr(outputs, 'rope_deltas'):
            output_dict['rope_deltas'] = outputs.rope_deltas
        
        # 添加位置预测相关输出
        output_dict['predicted_positions'] = predicted_positions
        output_dict['position_token_indices'] = position_token_indices
        
        return output_dict
    
    def generate(self, *args, **kwargs):
        """推理时使用基础模型的generate方法"""
        return self.base_model.generate(*args, **kwargs)
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """准备生成输入"""
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    
    def get_base_model(self):
        """获取基础模型，用于保存checkpoint"""
        return self.base_model
    
    def resize_token_embeddings(self, new_num_tokens):
        """调整token embeddings大小"""
        return self.base_model.resize_token_embeddings(new_num_tokens)
    
    def gradient_checkpointing_enable(self, **kwargs):
        """启用gradient checkpointing"""
        return self.base_model.gradient_checkpointing_enable(**kwargs)