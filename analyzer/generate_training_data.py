import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

from analyzer.nlp_processor import PolicyPredictionDataset
from utils.constants import PROCESSED_DATA_DIR
from utils.util import load_json
import sys



def train_t5_from_dataset(subdir: str, model_name: str = "t5-small", output_dir: str = "./t5_policy_output"):
    """
    하위 폴더 안의 JSON 파일을 자동으로 찾아 T5 모델 학습을 수행하는 함수
    """

    folder_path = os.path.join(PROCESSED_DATA_DIR, subdir)

    # 폴더 내 JSON 파일 탐색
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if not json_files:
        print(f"❌ JSON 파일을 찾을 수 없습니다: {folder_path}")
        return

    # 가장 첫 번째 JSON 파일을 선택 (또는 원하는 기준 정렬 가능)
    json_file = json_files[0]
    dataset_path = os.path.join(folder_path, json_file)
    print(f"📦 데이터 로딩 중: {dataset_path}")

    raw_data = load_json(dataset_path)
    if not raw_data:
        print("❌ 데이터가 없거나 파일 로딩에 실패했습니다.")
        return

    # 모델 및 토크나이저 로드
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # 학습/검증 분리
    train_data, val_data = train_test_split(raw_data, test_size=0.1, random_state=42)

    # 데이터셋 구성
    train_dataset = PolicyPredictionDataset(train_data, tokenizer)
    val_dataset = PolicyPredictionDataset(val_data, tokenizer)

    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=100,
        logging_steps=50,
        save_steps=200,
        evaluation_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available()
    )

    # Trainer 실행
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("🚀 학습 시작...")
    trainer.train()
    print("✅ 학습 완료.")


if __name__ == "__main__":
    # 예시: data/processed/1976 폴더 내 JSON 파일 자동 선택
    train_t5_from_dataset(subdir="1976")
