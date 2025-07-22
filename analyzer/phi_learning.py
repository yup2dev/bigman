MODEL_NAME = "microsoft/phi-2"

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

dataset = load_dataset("json", data_files="../crawler/trump_basis_prompts.jsonl", split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    torch_dtype="auto",
    device_map="auto",
    llm_int8_enable_fp32_cpu_offload=True
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
        max_length=512,
        padding="max_length",
        return_tensors=None,
    )
    out["labels"] = out["input_ids"].copy()
    return out

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=2,
    learning_rate=1e-4,
    output_dir="./phi2_trump_lora",
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
model.save_pretrained("./phi2_trump_lora")
tokenizer.save_pretrained("./phi2_trump_lora")
