import pandas as pd
import logging
import torch
import time
import os
import nltk
import numpy as np
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    DataCollatorForSeq2Seq
from multiprocessing import freeze_support

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='모델 학습 설정')
    parser.add_argument('--use_gpu', action='store_true', help='GPU 사용 여부 (기본값: False)')
    parser.add_argument('--use_amd', action='store_true', help='AMD GPU 사용 여부 (기본값: False)')
    parser.add_argument('--target_person', type=str, default='Donald Trump', help='중심 인물 (기본값: Donald Trump)')
    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    logger = logging.getLogger(__name__)

    def setup_device():
        """GPU 설정을 초기화하고 사용 가능한 디바이스를 반환합니다."""
        if args.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU 사용 설정: {gpu_name}")
            logger.info(f"CUDA 버전: {torch.version.cuda}")
            logger.info(f"사용 가능한 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # CUDA 메모리 최적화 설정
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
        elif args.use_amd and os.environ.get('ROCM_HOME') is not None:
            device = torch.device("cuda")
            logger.info("AMD GPU 사용 설정")
            logger.info(f"ROCm 버전: {os.environ.get('ROCM_HOME')}")
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
        else:
            device = torch.device("cpu")
            if args.use_gpu:
                logger.info("GPU 사용이 요청되었지만 사용할 수 없습니다. CPU로 실행됩니다.")
                logger.info("사용 가능한 GPU가 없거나 CUDA가 설치되지 않았습니다.")
            elif args.use_amd:
                logger.info("AMD GPU 사용이 요청되었지만 사용할 수 없습니다. CPU로 실행됩니다.")
                logger.info("ROCm이 설치되지 않았거나 AMD GPU가 감지되지 않았습니다.")
            else:
                logger.info("CPU 사용 설정")
        
        return device

    # 디바이스 설정
    device = setup_device()

    try:
        # 1. 데이터셋 로드
        logger.info("데이터셋 로드 중...")
        csv_path = 'data/cause_effect_dataset.csv'
        if not os.path.exists(csv_path):
            logger.error(f"CSV 파일이 존재하지 않습니다: {csv_path}")
            raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"로드된 데이터 크기: {len(df)}개 샘플")
        logger.info(f"데이터 미리보기:\n{df.head()}")

        # 결측치 제거
        df = df.dropna()
        logger.info(f"결측치 제거 후 데이터 크기: {len(df)}개 샘플")
        if len(df) == 0:
            logger.error("유효한 데이터가 없습니다.")
            raise ValueError("유효한 데이터가 없습니다.")

        # 2. 입력 텍스트와 타겟 텍스트 생성
        logger.info("데이터 전처리 중...")
        df['input_text'] = df.apply(
            lambda
                row: f"extract cause-effect for {row['person']} in context: {row['context']} | cause_time: {row['cause_time']} | effect_time: {row['effect_time']}",
            axis=1
        )
        df['target_text'] = df.apply(
            lambda
                row: f"person: {row['person']} | cause: {row['cause']} | cause_time: {row['cause_time']} | effect: {row['effect']} | effect_time: {row['effect_time']}",
            axis=1
        )

        # 3. Hugging Face Dataset 변환
        logger.info("Dataset 변환 중...")
        dataset = Dataset.from_pandas(df[['input_text', 'target_text']])
        logger.info(f"변환된 데이터셋 크기: {len(dataset)}개 샘플")

        # 데이터셋 분할 (크기에 따라 조정)
        if len(dataset) > 1:
            dataset = dataset.train_test_split(test_size=0.2, seed=42)
            train_dataset = dataset['train']
            eval_dataset = dataset['test']
            logger.info(f"훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(eval_dataset)}개")
        else:
            train_dataset = dataset
            eval_dataset = dataset  # 검증 데이터 없음
            logger.info(f"데이터셋 크기가 작아 분할하지 않음. 훈련 데이터: {len(train_dataset)}개")

        # 4. 토크나이저 및 모델 준비
        logger.info("모델 로딩 중...")
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

        # 5. 전처리 함수
        def preprocess(examples):
            inputs = tokenizer(
                examples['input_text'],
                max_length=512,
                truncation=True,
                padding=True
            )
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
        logger.info("데이터셋 전처리 중...")
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

        # 7. 데이터 콜레이터 설정
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )

        # 8. 학습 인자 설정
        learning_rate = 5e-5
        batch_size = 2
        gradient_accumulation = 4
        num_workers = 0
        fp16 = device.type == 'cuda'
        training_args = Seq2SeqTrainingArguments(
            output_dir="./cause_effect_model",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            num_train_epochs=10,
            weight_decay=0.01,
            predict_with_generate=True,
            logging_dir='./logs',
            save_strategy='epoch',
            logging_steps=10,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            dataloader_num_workers=num_workers,
            fp16=fp16,
            save_total_limit=2
        )

        # 9. 트레이너 설정
        logger.info("학습 시작...")
        
        def compute_metrics(eval_pred):
            """평가 메트릭을 계산하는 함수"""
            predictions, labels = eval_pred
            # 예측과 레이블의 길이를 맞추기 위해 패딩
            max_len = max(predictions.shape[1], labels.shape[1])
            padded_predictions = np.pad(predictions, ((0, 0), (0, max_len - predictions.shape[1])), mode='constant', constant_values=tokenizer.pad_token_id)
            padded_labels = np.pad(labels, ((0, 0), (0, max_len - labels.shape[1])), mode='constant', constant_values=tokenizer.pad_token_id)
            
            # 패딩된 토큰을 무시하고 손실 계산
            mask = (padded_labels != tokenizer.pad_token_id)
            loss = ((padded_predictions - padded_labels) ** 2 * mask).sum() / mask.sum()
            return {"eval_loss": float(loss)}

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        # 10. 학습 실행
        start_time = time.time()
        train_result = trainer.train()
        end_time = time.time()
        
        # 학습 결과 상세 출력
        logger.info(f"\n학습 완료! (소요 시간: {(end_time - start_time)/60:.1f}분)")
        logger.info(f"최종 손실: {train_result.training_loss:.4f}")
        logger.info(f"전체 스텝 수: {train_result.global_step}")
        logger.info(f"에포크 수: {training_args.num_train_epochs}")
        
        # 최종 평가
        logger.info("최종 평가 중...")
        eval_results = trainer.evaluate()
        logger.info(f"최종 검증 손실: {eval_results['eval_loss']:.4f}")

        # 11. 모델 저장
        logger.info("모델 저장 중...")
        trainer.save_model("./cause_effect_model")
        tokenizer.save_pretrained("./cause_effect_model")
        logger.info("모델 저장 완료!")

    except Exception as e:
        logger.error(f"에러 발생: {str(e)}")
        raise


if __name__ == '__main__':
    freeze_support()
    main()