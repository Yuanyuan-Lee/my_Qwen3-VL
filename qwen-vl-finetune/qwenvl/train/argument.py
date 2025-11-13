from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_position_head: bool = field(
        default=True,
        metadata={"help": "Whether to use position prediction head"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data (json file)"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the evaluation data (json file)"}
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Folder containing images"}
    )
    model_type: str = field(
        default="qwen3vl",
        metadata={"help": "Model type: qwen2vl, qwen2.5vl, qwen3vl"}
    )
    
    # 数据集相关
    dataset_use: str = field(
        default="qwen_stack_bowls_8_v3%100",
        metadata={"help": "Dataset to use with percentage, e.g., dataset_name%100"}
    )
    data_flatten: bool = field(
        default=True,
        metadata={"help": "Whether to flatten the data structure"}
    )
    data_packing: bool = field(
        default=False,
        metadata={"help": "Whether to pack multiple samples into one sequence"}
    )
    
    # 位置预测相关
    total_segments: int = field(
        default=8,
        metadata={"help": "Number of position segments to predict"}
    )
    segment_size: int = field(
        default=8,
        metadata={"help": "Frames per segment"}
    )
    total_frames: int = field(
        default=64,
        metadata={"help": "Total frames for trajectory prediction"}
    )
    
    # 数据预处理
    lazy_preprocess: bool = field(
        default=True,
        metadata={"help": "Whether to use lazy preprocessing"}
    )
    is_multimodal: bool = field(
        default=True,
        metadata={"help": "Whether the task is multimodal"}
    )
    image_aspect_ratio: str = field(
        default="pad",
        metadata={"help": "Image aspect ratio handling: pad, resize, crop"}
    )
    
    # 图像处理参数
    max_pixels: int = field(
        default=50176,
        metadata={"help": "Maximum number of pixels for image processing"}
    )
    min_pixels: int = field(
        default=784,
        metadata={"help": "Minimum number of pixels for image processing"}
    )
    
    # 视频处理参数
    video_min_frames: int = field(
        default=1,
        metadata={"help": "Minimum number of frames for video processing"}
    )
    video_max_frames: int = field(
        default=64,
        metadata={"help": "Maximum number of frames for video processing"}
    )
    video_fps: int = field(
        default=1,
        metadata={"help": "Frames per second for video sampling"}
    )
    video_min_pixels: int = field(
        default=784,
        metadata={"help": "Minimum number of pixels for video frame processing"}
    )
    video_max_pixels: int = field(
        default=50176,
        metadata={"help": "Maximum number of pixels for video frame processing"}
    )
    
    # 其他可能需要的参数
    image_size: int = field(
        default=448,
        metadata={"help": "Image size for preprocessing"}
    )
    pad_image_to_square: bool = field(
        default=True,
        metadata={"help": "Whether to pad images to square"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length"}
    )
    
    # 位置预测相关参数
    position_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for position prediction loss"}
    )
    save_full_model: bool = field(
        default=False,
        metadata={"help": "Whether to save full model including position head"}
    )
    
    # 模型微调策略
    tune_mm_vision: bool = field(
        default=False,
        metadata={"help": "Whether to tune the vision tower"}
    )
    tune_mm_mlp: bool = field(
        default=True,
        metadata={"help": "Whether to tune the MLP projector"}
    )
    tune_mm_llm: bool = field(
        default=True,
        metadata={"help": "Whether to tune the language model"}
    )
    
    # Qwen3VL特定参数
    gradient_checkpointing: bool = field(default=True)
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    
    # LoRA参数（如果使用）
    use_lora: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )
    
    # 训练参数
    remove_unused_columns: bool = field(default=False)
    freeze_vision_tower: bool = field(default=False)
    freeze_language_model: bool = field(default=False)
    
    # 学习率调度
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.03)
    
    # 保存和日志
    logging_steps: int = field(default=1)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    evaluation_strategy: str = field(default="no")
    
    # 分布式训练
    ddp_find_unused_parameters: bool = field(default=False)
    ddp_backend: str = field(default="nccl")
    
    # 梯度相关
    gradient_accumulation_steps: int = field(default=1)
    max_grad_norm: float = field(default=1.0)
    
    # Deepspeed
    deepspeed: Optional[str] = field(default=None)
    
    # 其他
    report_to: str = field(default="tensorboard")
    dataloader_pin_memory: bool = field(default=True)
    group_by_modality_length: bool = field(default=True)
