# -*- coding: utf-8 -*-
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import time

# 检测 flash-attn 是否可用（放在文件顶部靠近其它 import 之后）
try:
    import flash_attn
    print("flash_attn available:", getattr(flash_attn, "__version__", "unknown"))
except Exception as e:
    print("flash_attn NOT available:", e)

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


if __name__ == '__main__':
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
    #             },
    #             {"type": "text", "text": "这段视频有多长"},
    #         ],
    #     }
    # ]

    messages = [
        {
            "role": "user",
            "content": [
              {
                  "type": "image",
                  "image": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png",
              },
              {"type": "text", "text": "Read all the text in the image."},
            ],
        }
    ]

    # TODO: change to your own checkpoint path
    checkpoint_path = "/share/project/liyuanyuan/code/Qwen3-VL/weights/Qwen3-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    inputs = [prepare_inputs_for_vllm(message, processor) for message in [messages]]

    llm = LLM(
        model=checkpoint_path,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=False,  # 关闭 MoE 专家并行（若使用 MoE 模型则改为 True 并提供正确的专家配置）
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        top_k=-1,
        stop_token_ids=[],
    )

    for i, input_ in enumerate(inputs):
        print()
        print('=' * 40)
        print(f"Inputs[{i}]: {input_['prompt']=!r}")
    print('\n' + '>' * 40)

    # 同步辅助函数（若使用 GPU 则确保 CUDA 操作完成以测量真实耗时）
    def _sync_cuda():
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.synchronize()

    # 统计推理时间
    total_start = time.perf_counter()
    _sync_cuda()
    infer_start = time.perf_counter()

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    _sync_cuda()
    infer_end = time.perf_counter()
    total_end = time.perf_counter()

    infer_time = infer_end - infer_start
    total_time = total_end - total_start

    # 输出结果并统计吞吐（用字符数作为近似）
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        length_chars = len(generated_text)
        print()
        print('=' * 40)
        print(f"Generated text: {generated_text!r}")
        print(f"Output[{i}] 字符数: {length_chars}")

    print('\n' + '-' * 40)
    print(f"推理时间（仅 model.generate 调用）: {infer_time:.4f} 秒")
    print(f"总耗时（包括前后同步等）: {total_time:.4f} 秒")

    # 计算简单吞吐：总生成字符数 / 推理时间
    total_chars = sum(len(o.outputs[0].text) for o in outputs)
    if infer_time > 0:
        print(f"总生成字符数: {total_chars}, 吞吐(字符/秒): {total_chars / infer_time:.2f}")
    else:
        print("推理时间过短，无法计算吞吐。")