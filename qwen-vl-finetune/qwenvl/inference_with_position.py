import torch
from transformers import AutoProcessor, AutoTokenizer
import numpy as np
from PIL import Image

# 尝试导入Qwen3VL类
try:
    from transformers import Qwen3VLForConditionalGeneration
    MODEL_CLASS = Qwen3VLForConditionalGeneration
except ImportError:
    try:
        from transformers import Qwen2VLForConditionalGeneration as MODEL_CLASS
    except ImportError:
        from transformers import AutoModelForVision2Seq as MODEL_CLASS

try:
    from transformers import Qwen3VLProcessor
    PROCESSOR_CLASS = Qwen3VLProcessor
except ImportError:
    PROCESSOR_CLASS = AutoProcessor

from qwenvl.train.qwen_with_position_head import Qwen3VLWithPositionHead
from qwenvl.train.position_predictor import PositionPredictor

def load_model_for_inference(model_path, use_position_head=False, device="cuda"):
    """加载Qwen3VL模型用于推理"""
    try:
        processor = PROCESSOR_CLASS.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
    except:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    
    if use_position_head:
        # 加载完整模型（包括position head）
        model = Qwen3VLWithPositionHead(
            base_model_name_or_path=model_path,
            num_segments=8,
            position_dim=6,
            use_position_head=True
        )
        full_model_path = f"{model_path}/full_model.pt"
        model.load_state_dict(torch.load(full_model_path, map_location=device))
        print(f"Loaded full model with position head from {full_model_path}")
    else:
        # 只加载基础模型
        base_model_path = f"{model_path}/base_model" if "/base_model" not in model_path else model_path
        model = MODEL_CLASS.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        print(f"Loaded base model from {base_model_path}")
    
    model.eval()
    return model, processor

def predict_positions(
    model, 
    processor, 
    images,  # List of PIL Images or paths
    instruction, 
    history_endposes,
    device="cuda"
):
    """
    预测3D位置
    
    Args:
        images: 三个视角的图像 [left_camera, head_camera, right_camera]
        instruction: 指令文本
        history_endposes: 历史endpose数据 (3, 14)
    
    Returns:
        positions: numpy array of shape (num_segments, 6) 真实物理坐标
        position_text: 原始文本输出
    """
    # 加载图像
    image_list = []
    for img in images:
        if isinstance(img, str):
            image_list.append(Image.open(img).convert('RGB'))
        elif isinstance(img, Image.Image):
            image_list.append(img.convert('RGB'))
        else:
            image_list.append(img)
    
    # 构建prompt
    endpose_txt = np.array2string(
        history_endposes, 
        separator=', ',
        formatter={'float_kind': lambda x: f"{x:.6f}"}
    ).replace('\n', '')
    
    # Qwen3VL的消息格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_list[0]},
                {"type": "text", "text": "This is the left camera image of the current frame."},
                {"type": "image", "image": image_list[1]},
                {"type": "text", "text": "This is the head camera image of the current frame."},
                {"type": "image", "image": image_list[2]},
                {"type": "text", "text": "This is the right camera image of the current frame."},
                {
                    "type": "text",
                    "text": f"You are a Aloha-AgileX robot using end-effector control. "
                            f"The instruction is \"{instruction}\". "
                            f"The format of endpose is [x_l, y_l, z_l, roll_l, pitch_l, yaw_l, gripper_l, "
                            f"x_r, y_r, z_r, roll_r, pitch_r, yaw_r, gripper_r] and the previous three "
                            f"(including current) frames' endposes are: {endpose_txt}. "
                            f"Please predict the next 8 3D gripper positions at frames 8,16,24,32,40,48,56,64. "
                            f"Each position should contain [x_left, y_left, z_left, x_right, y_right, z_right]. "
                            f"The ranges are: x_left∈[-0.5,0], x_right∈[0,0.5], y_left,y_right∈[0,0.5], z_left,z_right∈[0,1]. "
                            f"Output format: <pos>x1,y1,z1,x2,y2,z2</pos>_<pos>x1,y1,z1,x2,y2,z2</pos>_..."
                }
            ]
        }
    ]
    
    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], 
        images=image_list, 
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None
        )
    
    # 解码输出
    generated_ids = output_ids[0][len(inputs.input_ids[0]):]
    output_text = processor.decode(generated_ids, skip_special_tokens=False)
    
    # 解析位置
    predictor = PositionPredictor()
    normalized_positions = predictor.parse_position_labels(output_text)
    
    if normalized_positions is not None:
        # 反归一化
        real_positions = predictor.denormalize_positions(normalized_positions)
        return real_positions.numpy(), output_text
    
    return None, output_text

# 使用示例
if __name__ == "__main__":
    model_path = "/path/to/trained/model"
    model, processor = load_model_for_inference(model_path, use_position_head=False)
    
    # 加载图像
    images = [
        "left_camera.jpg",
        "head_camera.jpg",
        "right_camera.jpg"
    ]
    
    # 历史endpose
    history_endposes = np.random.randn(3, 14)  # 示例数据
    
    # 预测
    positions, text = predict_positions(
        model, processor, images,
        instruction="Pick up the cup",
        history_endposes=history_endposes
    )
    
    if positions is not None:
        print("Predicted positions shape:", positions.shape)
        print("Predicted positions:\n", positions)
    else:
        print("Failed to parse positions")
    print("\nOutput text:", text)