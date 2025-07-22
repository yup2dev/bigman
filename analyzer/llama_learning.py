from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

MODEL_NAME = "NousResearch/Llama-2-7b-hf"     # 모델명 확인 (허깅페이스 경로)
DATA_PATH = "../crawler/trump_basis_prompts.jsonl"     # 학습 데이터 경로
OUTPUT_DIR = "./llama_trump_lora"           # 결과 저장 경로

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def tokenize(batch):
    out = tokenizer(
        batch["prompt"],
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors=None,
    )
    out["labels"] = out["input_ids"].copy()
    return out

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    per_device_train_batch_size=2,
    num_train_epochs=2,
    learning_rate=1e-4,
    output_dir=OUTPUT_DIR,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
