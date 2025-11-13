# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
import pathlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

# 尝试导入Qwen3VL相关的类
try:
    from transformers import Qwen3VLProcessor
    PROCESSOR_CLASS = Qwen3VLProcessor
except ImportError:
    PROCESSOR_CLASS = AutoProcessor

# 导入自定义模块
from qwenvl.train.qwen_with_position_head import Qwen3VLWithPositionHead
from qwenvl.train.trainer import Qwen3VLTrainer
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments as CustomTrainingArguments
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.data.data_collator_fix import DataCollatorForQwen3VL

# 设置日志
logger = logging.getLogger(__name__)

def train():
    # 1. 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    # 2. 加载processor和tokenizer
    logger.info(f"Loading processor and tokenizer from {model_args.model_name_or_path}")
    
    try:
        processor = PROCESSOR_CLASS.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True
        )
    except Exception as e:
        logger.warning(f"Failed to load with specific processor: {e}")
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True
        )
    
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    
    # 添加特殊token用于位置预测
    special_tokens_dict = {}
    if "<|position|>" not in tokenizer.get_vocab():
        special_tokens_dict["additional_special_tokens"] = ["<|position|>"]
    
    if len(special_tokens_dict) > 0:
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_new_tokens} special tokens to tokenizer")
    
    # 获取position token的ID
    position_token_id = tokenizer.convert_tokens_to_ids("<|position|>")
    logger.info(f"Position token '<|position|>' has ID: {position_token_id}")
    
    # 3. 创建带position head的Qwen3VL模型
    logger.info("Loading Qwen3VL model with position prediction head...")
    model = Qwen3VLWithPositionHead(
        base_model_name_or_path=model_args.model_name_or_path,
        num_segments=data_args.total_segments,
        position_dim=6,
        use_position_head=model_args.use_position_head
    )
    
    # 调整模型embedding以匹配新的tokenizer大小
    if len(special_tokens_dict) > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings to {len(tokenizer)}")
    
    # 设置position token ID
    if model_args.use_position_head:
        model.set_position_token_id(position_token_id)
    
    # 4. 启用gradient checkpointing（如果需要）
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # 5. 准备数据
    logger.info("Loading and processing dataset...")
    data_module = make_supervised_data_module(
        processor=processor, 
        data_args=data_args
    )
    
    # 替换 data collator
    data_module['data_collator'] = DataCollatorForQwen3VL(processor=processor)
    
    # ========== 验证数据 ==========
    logger.info("Validating dataset...")
    train_dataset = data_module['train_dataset']
    
    # 检查前5个样本
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        logger.info(f"\nSample {i}:")
        logger.info(f"  Keys: {sample.keys()}")
        
        if 'position_labels' in sample:
            logger.info(f"  ✓ Has position_labels: {sample['position_labels']}")
        else:
            logger.warning(f"  ✗ Missing position_labels!")
    
    # 检查一个 batch
    collator = data_module['data_collator']
    batch = collator([train_dataset[i] for i in range(4)])
    logger.info(f"\nBatch keys: {batch.keys()}")
    
    if 'position_labels' in batch:
        logger.info(f"  ✓ Batch has position_labels: shape={batch['position_labels'].shape}")
    else:
        logger.error(f"  ✗ Batch missing position_labels!")
    
    # 6. 创建Trainer
    logger.info("Initializing trainer...")
    trainer = Qwen3VLTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        position_loss_weight=training_args.position_loss_weight,
        **data_module
    )
    
    # 7. 训练
    logger.info("Starting training...")
    checkpoint_path = None
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        checkpoint_path = training_args.output_dir
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    
    if checkpoint_path:
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()
    
    # 8. 保存模型
    logger.info("Saving model...")
    trainer.save_state()
    
    # 保存基础模型（不包括position head）用于推理
    output_dir = training_args.output_dir
    base_model_dir = os.path.join(output_dir, "base_model")
    os.makedirs(base_model_dir, exist_ok=True)
    
    model.get_base_model().save_pretrained(base_model_dir)
    processor.save_pretrained(base_model_dir)
    logger.info(f"Saved base model to {base_model_dir}")
    
    # 如果需要保存完整模型（包括position head）
    if training_args.save_full_model:
        full_model_path = os.path.join(output_dir, "full_model.pt")
        torch.save(model.state_dict(), full_model_path)
        logger.info(f"Saved full model (with position head) to {full_model_path}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    train()
