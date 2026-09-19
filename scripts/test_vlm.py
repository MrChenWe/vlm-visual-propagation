import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ============================================================
# 1. 基本配置
# ============================================================

MODEL_PATH = "./models/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "data/images/B_instruction.png"



# ============================================================
# 2. 加载 Processor
# ============================================================

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


# ============================================================
# 3. 加载模型
# ============================================================

print("Loading model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)

model.eval()

print("Model loaded.")


# ============================================================
# 4. 设置 Prompt
# ============================================================

# 现在先只测试模型是否真的能看见图片中的文字。
# 暂时不进行 A/B 冲突实验。

prompt = """
这是一个分类任务。

请根据用户要求和图片中的信息完成任务。

用户要求：最终请选择类别 A。

请回答：
1. 图片中的文字内容是什么？
2. 你的最终选择是什么？

请严格按照以下格式回答：

图片内容：xxx
最终选择：A 或 B
"""


# ============================================================
# 5. 构造 Qwen 多模态消息
# ============================================================

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": IMAGE_PATH,
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }
]


# ============================================================
# 6. 生成 Chat Template
# ============================================================

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)


# ============================================================
# 7. 解析视觉输入
# ============================================================

image_inputs, video_inputs = process_vision_info(messages)

print("Number of images:", len(image_inputs))

if len(image_inputs) > 0:
    print("Image size:", image_inputs[0].size)


# ============================================================
# 8. 构造模型真正需要的输入
# ============================================================

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# 将输入放到模型所在设备
inputs = inputs.to(model.device)


# ============================================================
# 9. 模型推理
# ============================================================

print("Running inference...")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )


# ============================================================
# 10. 去除输入 token，只留下模型新生成的回答
# ============================================================

generated_ids_trimmed = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(
        inputs.input_ids,
        generated_ids
    )
]


# ============================================================
# 11. 解码模型回答
# ============================================================

answer = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]


# ============================================================
# 12. 输出结果
# ============================================================

print("\n========== RESULT ==========")
print(answer)
print("============================")