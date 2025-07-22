from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

BASE_MODEL = "microsoft/phi-2"          # LoRA 튜닝에 사용한 base 모델명
LORA_DIR = "./phi2_trump_lora"          # LoRA 결과 폴더
OFFLOAD_DIR = "./offload_dir"           # 오프로드 폴더

# 1. 모델/LoRA 로드
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="auto",
    offload_folder=OFFLOAD_DIR
)
model = PeftModel.from_pretrained(
    base_model,
    LORA_DIR,
    offload_folder=OFFLOAD_DIR
)

# 2. 테스트 프롬프트(질문) 예시
prompt = """
[Episode ID: test_20240723 | Date: 2024-07-23 | Type: hypothetical interview]

Q: If New York's property market crashes again, how would you act?

Trump's answer:
"""

# 3. 텍스트 생성
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generate_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
output = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

print("\n===== 트럼프 사고 흐름 답변 =====\n")
print(output)
