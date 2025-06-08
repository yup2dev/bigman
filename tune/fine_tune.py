import os
import json
import torch
import logging
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, DatasetDict

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "tune", "bart_cause_effect_model")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def load_json_data(data_root: str) -> list:
    """
    data_root 하위의 모든 폴더 내 JSON 파일을 재귀적으로 탐색하여
    학습용 input_text / target_text 샘플을 추출합니다.
    """
    samples = []

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {data_root}")

    # 재귀 탐색으로 모든 .json 파일 수집
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        if not isinstance(data, list):
                            logger.warning(f"⚠️ {file}은 리스트가 아닙니다. 건너뜁니다.")
                            continue

                        for item in data:
                            conditions = " ".join(item.get("parsed", {}).get("conditions", []))
                            conclusion = item.get("parsed", {}).get("conclusion", "")
                            effects = item.get("expected_effects", [])
                            impact_text = " | ".join(effects) if effects else "N/A"

                            input_text = f"{conditions} → {conclusion}"
                            target_text = f"effect: {impact_text}"

                            samples.append({
                                "input_text": input_text,
                                "target_text": target_text
                            })

                except Exception as ex:
                    logger.warning(f"⚠️ {file} 처리 중 오류 발생: {ex}")

    return samples



def tokenize_data(dataset: DatasetDict, tokenizer: BartTokenizer) -> DatasetDict:
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

    logger.info("✏️ 토크나이징 중...")
    return dataset.map(tokenize_function, batched=True)


def main():
    logger.info("📂 데이터 로딩 시작...")
    data = load_json_data(DATA_DIR)
    logger.info(f"✅ 총 샘플 수: {len(data)}")

    if not data:
        logger.error("❌ 학습할 데이터가 없습니다.")
        return

    raw_dataset = Dataset.from_list(data)

    # 샘플 수에 따라 분할 여부 결정
    if len(raw_dataset) > 1:
        dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        logger.warning("⚠️ 샘플 수가 1개이므로 train/test 분할을 건너뜁니다.")
        train_dataset = raw_dataset
        eval_dataset = raw_dataset  # 또는 None

    # 모델/토크나이저 로드
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # 토크나이징
    tokenized_datasets = tokenize_data(DatasetDict({
        "train": train_dataset,
        "test": eval_dataset
    }), tokenizer)

    # 학습 인자 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir=os.path.join(MODEL_SAVE_DIR, "logs"),
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=torch.cuda.is_available()
    )

    # Trainer 구성
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
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
