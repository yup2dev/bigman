import os
import json
import torch
import logging
from datetime import datetime
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 경로 설정
DATE = "2025-04-17"
DATA_DIR = os.path.join("analyzer", "data", "processed", DATE)
MODEL_SAVE_DIR = os.path.join("tune", "bart_cause_effect_model")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def load_json_data(data_dir):
    samples = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            path = os.path.join(data_dir, file)
            with open(path, 'r', encoding='utf-8') as f:
                article = json.load(f)
                for e in article.get("events", []):
                    input_text = f"{e['person']} in context: {e['context']} → {e['decision']}"
                    target_text = f"effect: {e['outcome']} | impact: {e['impact_type']}"
                    samples.append({
                        "input_text": input_text,
                        "target_text": target_text
                    })
    return samples

def main():
    logger.info("📂 데이터 로딩 시작...")
    data = load_json_data(DATA_DIR)
    logger.info(f"✅ 총 샘플 수: {len(data)}")

    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 모델/토크나이저
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    def tokenize_function(example):
        model_inputs = tokenizer(
            example["input_text"], max_length=512, padding="max_length", truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example["target_text"], max_length=128, padding="max_length", truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info("✏️ 토크나이즈 중...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # 학습 설정
    args = Seq2SeqTrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        predict_with_generate=True,
        fp16=torch.cuda.is_available()
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    logger.info("🚀 학습 시작...")
    trainer.train()

    logger.info("💾 모델 저장 중...")
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    logger.info("🎉 완료!")

if __name__ == "__main__":
    main()
