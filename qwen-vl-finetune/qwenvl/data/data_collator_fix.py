import torch
from dataclasses import dataclass
from typing import Dict, Sequence, List
import transformers

@dataclass
class DataCollatorForQwen3VL:
    """
    修复版的 Data Collator for Qwen3VL - 确保 image_grid_thw 的正确格式
    """
    processor: transformers.ProcessorMixin
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 提取并转换 input_ids 为 list
        input_ids = []
        for instance in instances:
            ids = instance['input_ids']
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(ids, list) and len(ids) > 0:
                if isinstance(ids[0], list):
                    ids = ids[0]
            input_ids.append(ids)
        
        # 提取 labels
        labels = []
        for instance in instances:
            lbls = instance['labels']
            if isinstance(lbls, torch.Tensor):
                lbls = lbls.tolist()
            if isinstance(lbls, list) and len(lbls) > 0:
                if isinstance(lbls[0], list):
                    lbls = lbls[0]
            labels.append(lbls)
        
        # 找到最大长度
        max_length = max(len(ids) for ids in input_ids)
        
        # 获取 pad token
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        
        # 手动 padding
        input_ids_padded = []
        attention_mask = []
        for ids in input_ids:
            padding_length = max_length - len(ids)
            padded_ids = ids + [pad_token_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length
            input_ids_padded.append(padded_ids)
            attention_mask.append(mask)
        
        # 手动 padding labels
        labels_padded = []
        for lbls in labels:
            padding_length = max_length - len(lbls)
            padded_labels = lbls + [-100] * padding_length
            labels_padded.append(padded_labels)
        
        # 转为 tensor
        batch = {
            'input_ids': torch.tensor(input_ids_padded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels_padded, dtype=torch.long)
        }
        
        # 处理 pixel_values
        if 'pixel_values' in instances[0] and instances[0]['pixel_values'] is not None:
            pixel_values_list = []
            for instance in instances:
                pv = instance['pixel_values']
                if isinstance(pv, torch.Tensor):
                    pixel_values_list.append(pv)
                else:
                    pixel_values_list.append(torch.tensor(pv))
            
            if len(pixel_values_list) > 0:
                try:
                    batch['pixel_values'] = torch.stack(pixel_values_list)
                except RuntimeError as e:
                    print(f"Warning: Cannot stack pixel_values: {e}")
                    batch['pixel_values'] = pixel_values_list[0].unsqueeze(0)
        
        # 处理 image_grid_thw - 使用 cat 展平所有图像
        if 'image_grid_thw' in instances[0] and instances[0]['image_grid_thw'] is not None:
            image_grid_thw_list = []
            for instance in instances:
                grid = instance['image_grid_thw']
                
                # 转换为 Python list/integers
                if isinstance(grid, torch.Tensor):
                    grid = grid.cpu().numpy().tolist()
                
                # 确保是列表格式
                if not isinstance(grid, list):
                    grid = [grid]
                
                # 递归展平嵌套列表
                def flatten(x):
                    if isinstance(x, (list, tuple)):
                        result = []
                        for item in x:
                            if isinstance(item, (list, tuple)):
                                result.extend(flatten(item))
                            else:
                                result.append(int(item))
                        return result
                    else:
                        return [int(x)]
                
                grid_flat = flatten(grid)
                
                # image_grid_thw 应该是 [num_images, 3] 的格式
                # 每3个值组成一组 (t, h, w)
                if len(grid_flat) % 3 != 0:
                    # print(f"Warning: image_grid_thw length not divisible by 3: {len(grid_flat)}, data: {grid_flat}")
                    # 补齐或截断
                    if len(grid_flat) < 3:
                        grid_flat.extend([1] * (3 - len(grid_flat)))
                    else:
                        grid_flat = grid_flat[:len(grid_flat) // 3 * 3]
                
                # 重塑为 [num_images, 3]
                num_images = len(grid_flat) // 3
                grid_reshaped = []
                for i in range(num_images):
                    grid_reshaped.append([
                        int(grid_flat[i*3]),
                        int(grid_flat[i*3+1]), 
                        int(grid_flat[i*3+2])
                    ])
                
                # 转为 tensor 并添加到列表
                grid_tensor = torch.tensor(grid_reshaped, dtype=torch.long)
                image_grid_thw_list.append(grid_tensor)
            
            if len(image_grid_thw_list) > 0:
                try:
                    # 使用 cat 将所有图像展平到第一维: [total_images, 3]
                    batch['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)
                except RuntimeError as e:
                    print(f"Warning: Cannot cat image_grid_thw: {e}")
                    # print(f"Shapes: {[g.shape for g in image_grid_thw_list]}")
                    batch['image_grid_thw'] = image_grid_thw_list[0]
        
        # 处理位置标签
        if 'position_labels' in instances[0] and instances[0]['position_labels'] is not None:
            position_labels_list = []
            for instance in instances:
                pos = instance['position_labels']
                pos = torch.tensor(pos, dtype=torch.float32)
                position_labels_list.append(pos)
            batch['position_labels'] = torch.stack(position_labels_list)
        
        # 调试代码 - 训练稳定后已注释
        # if 'image_grid_thw' in batch:
        #     grid = batch['image_grid_thw']
        #     print(f"Debug: image_grid_thw shape={grid.shape}, dtype={grid.dtype}")
        #     if len(grid) > 0:
        #         print(f"  First image grid: {grid[0]}")
        #         print(f"  Value types: t={type(grid[0, 0].item())}, h={type(grid[0, 1].item())}, w={type(grid[0, 2].item())}")
        
        return batch