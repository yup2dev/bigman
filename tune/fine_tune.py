import pandas as pd
import logging
import torch
import time
import os
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# 명령행 인자 파싱
parser = argparse.ArgumentParser(description='모델 학습 설정')
parser.add_argument('--use_gpu', action='store_true', help='GPU 사용 여부 (기본값: False)')
parser.add_argument('--use_amd', action='store_true', help='AMD GPU 사용 여부 (기본값: False)')
args = parser.parse_args()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"사용 가능한 디바이스: {device}")

# AMD GPU 설정
if torch.cuda.is_available():
    logger.info(f"GPU 정보: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    # AMD GPU를 위한 ROCm 설정 시도
    if os.environ.get('ROCM_HOME') is not None:
        logger.info("ROCm이 설치되어 있습니다.")
        # ROCm을 사용하도록 PyTorch 설정
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        device = torch.device("cuda")
        logger.info("AMD GPU를 사용하도록 설정되었습니다.")
    else:
        logger.warning("GPU를 사용할 수 없습니다. ROCm이 설치되어 있지 않거나 올바르게 설정되지 않았습니다.")
        logger.info("AMD GPU를 사용하려면 다음 단계를 수행하세요:")
        logger.info("1. AMD ROCm을 설치하세요: https://rocm.docs.amd.com/en/latest/Installation_Guide/Installation-Guide.html")
        logger.info("2. PyTorch ROCm 버전을 설치하세요: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2")
        logger.info("3. 환경 변수 ROCm_HOME이 올바르게 설정되어 있는지 확인하세요")

try:
    # 1. 데이터 로드
    logger.info("데이터 파일 로딩 시작...")
    df = pd.read_csv('data/cause_effect_dataset.csv')
    logger.info(f"데이터 로드 완료. 총 {len(df)}개의 샘플이 있습니다.")
    logger.info(f"데이터 컬럼: {df.columns.tolist()}")
    logger.info("\n데이터 샘플:")
    logger.info(df.head())

    # 2. 입력 텍스트와 타겟 텍스트 생성
    logger.info("입력 및 타겟 텍스트 생성 중...")
    df['input_text'] = df.apply(
        lambda row: f"extract cause-effect for {row['person']}: {row['context']}",
        axis=1
    )

    df['target_text'] = df.apply(
        lambda row: f"person: {row['person']} | cause: {row['cause']} | cause_time: {row['cause_time']} | effect: {row['effect']} | effect_time: {row['effect_time']}",
        axis=1
    )
    logger.info("텍스트 생성 완료")
    logger.info("\n생성된 입력 텍스트 샘플:")
    logger.info(df['input_text'].head())
    logger.info("\n생성된 타겟 텍스트 샘플:")
    logger.info(df['target_text'].head())

    # 3. Hugging Face Dataset 변환
    logger.info("Hugging Face Dataset으로 변환 중...")
    dataset = Dataset.from_pandas(df[['input_text', 'target_text']])
    
    # 데이터셋 분할 (80% 훈련, 20% 검증)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    logger.info(f"훈련 데이터셋 크기: {len(train_dataset)}")
    logger.info(f"검증 데이터셋 크기: {len(eval_dataset)}")
    logger.info(f"Dataset 변환 완료. 특징: {dataset['train'].features}")

    # 4. 토크나이저 및 모델 준비
    logger.info("토크나이저 및 모델 로딩 중...")
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    logger.info(f"모델 {model_name} 로딩 완료")

    # 5. 전처리 함수
    def preprocess(examples):
        logger.info(f"전처리 중: {len(examples['input_text'])}개 샘플")
        # 입력 텍스트 처리
        inputs = tokenizer(
            examples['input_text'],
            max_length=512,
            truncation=True,
            padding=True
        )

        # 타겟 텍스트 처리
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target_text'],
                max_length=128,
                truncation=True,
                padding=True
            )

        inputs['labels'] = labels['input_ids']
        return inputs

    # 6. 데이터셋 전처리
    logger.info("데이터셋 전처리 시작...")
    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        preprocess,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    logger.info("데이터셋 전처리 완료")

    # 7. 데이터 콜레이터 설정
    logger.info("데이터 콜레이터 설정 중...")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # 8. 학습 인자 설정
    logger.info("학습 인자 설정 중...")
    
    # GPU/CPU에 따른 최적 설정
    if torch.cuda.is_available() or os.environ.get('ROCM_HOME') is not None:
        # GPU 설정 (NVIDIA 또는 AMD)
        batch_size = 8
        gradient_accumulation = 2
        learning_rate = 2e-4
        fp16 = True
        num_workers = 4
        logger.info("GPU 설정을 사용합니다.")
    else:
        # CPU 설정
        batch_size = 2
        gradient_accumulation = 4
        learning_rate = 1e-4
        fp16 = False
        num_workers = 0
        logger.info("CPU 설정을 사용합니다.")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./cause_effect_model",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=5,
        weight_decay=0.01,
        predict_with_generate=True,
        logging_dir='./logs',
        save_strategy='epoch',
        logging_steps=10,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        dataloader_num_workers=num_workers,
        fp16=fp16
    )

    # 9. 트레이너 설정
    logger.info("트레이너 설정 중...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 검증 데이터셋 추가
        data_collator=data_collator
    )

    # 10. 학습 실행
    logger.info("학습 시작...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    
    # 학습 결과 분석
    logger.info("\n=== 학습 결과 분석 ===")
    logger.info(f"총 학습 시간: {end_time - start_time:.2f}초")
    logger.info(f"초당 처리 샘플 수: {len(train_dataset) / (end_time - start_time):.2f}")
    logger.info(f"최종 학습 손실: {train_result.training_loss:.4f}")
    
    # 메트릭 저장
    metrics = train_result.metrics
    logger.info("\n=== 상세 메트릭 ===")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")

    # 11. 모델 저장
    logger.info("모델 저장 중...")
    trainer.save_model("./cause_effect_model")
    tokenizer.save_pretrained("./cause_effect_model")
    logger.info("모델 저장 완료!")

except Exception as e:
    logger.error(f"에러 발생: {str(e)}", exc_info=True)
    raise