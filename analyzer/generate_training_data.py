import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utils.constants import PROCESSED_DATA_DIR
from utils.util import load_json

# 사용자 정의 데이터셋 클래스
class PolicySentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 감정 레이블을 숫자로 매핑 (NEUTRAL: 0, POSITIVE: 1, NEGATIVE: 2, MIXED: 3)
        self.label_map = {"NEUTRAL": 0, "POSITIVE": 1, "NEGATIVE": 2, "MIXED": 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        gpt_input = item.get('gpt_input', '')
        if not gpt_input:
            print(f"경고: gpt_input 없음 (idx={idx})")
            gpt_input = ""

        # 입력 전체를 BERT에 넣기
        encoding = self.tokenizer(
            gpt_input,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 레이블 매핑 (없으면 -100으로 무시)
        sentiment = item.get('gpt_sentiment_label', 'NEUTRAL')
        label = self.label_map.get(sentiment, 0)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),      # (max_length,)
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 학습 함수
def train_bert_from_multiple_years(start_year=1976, end_year=1990, model_name="bert-base-uncased", output_dir="./bert_policy_output"):
    # 데이터 읽는 부분 (사용자 코드에서 그대로 가져옴)
    all_data = []
    for year in range(start_year, end_year + 1):
        folder_path = os.path.join(PROCESSED_DATA_DIR, str(year))
        if not os.path.exists(folder_path):
            print(f"❌ 폴더 없음: {folder_path}")
            continue
        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        if not json_files:
            print(f"❌ JSON 파일 없음: {folder_path}")
            continue
        for json_file in json_files:
            dataset_path = os.path.join(folder_path, json_file)
            print(f"📦 {year}년 데이터 로딩 중: {dataset_path}")
            raw_data = load_json(dataset_path)
            if not raw_data:
                print(f"❌ 데이터 없음: {dataset_path}")
                continue
            all_data.extend(raw_data)

    if not all_data:
        print("❌ 전체 데이터를 합칠 수 없습니다.")
        return

    # 모델 및 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)

    # 학습/검증 데이터 분리
    train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)

    # 데이터셋 생성
    train_dataset = PolicySentimentDataset(train_data, tokenizer)
    val_dataset = PolicySentimentDataset(val_data, tokenizer)

    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        # evaluation_strategy="steps",
        fp16=torch.cuda.is_available()
    )

    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # 학습 시작
    print("🚀 전체 연도+파일 합산 데이터로 BERT 모델 학습 시작...")
    train_result = trainer.train()
    log_history = trainer.state.log_history

    # 손실 그래프 시각화
    df = pd.DataFrame(log_history)
    plt.figure(figsize=(8, 5))
    if 'loss' in df.columns:
        plt.plot(df['step'], df['loss'], label='train_loss')
    if 'eval_loss' in df.columns:
        plt.plot(df['step'], df['eval_loss'], label='eval_loss')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training & Evaluation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    print("✅ BERT 모델 학습 완료.")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_bert_from_multiple_years(
        start_year=1976,
        end_year=1990,
        model_name="bert-base-uncased",
        output_dir="./bert_policy_output"
    )