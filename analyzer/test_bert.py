import torch
from transformers import BertTokenizer, BertForSequenceClassification

def predict_sentiment(text, tokenizer, model, device, max_length=128):
    model.eval()
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
    return pred

def main():
    model_dir = "./bert_policy_output"  # 학습한 모델 폴더로 변경
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)

    label_map = label_map = {
                                0: "NEUTRAL",
                                1: "POSITIVE",
                                2: "NEGATIVE",
                                3: "MIXED"
                            }

    test_question = "You've been recognized for your bold investments in the city. However, some believe your aggressive approach has led to increased conflicts with local communities. What's your take?"
    pred = predict_sentiment(test_question, tokenizer, model, device)
    print(f"Input: {test_question}")
    print(f"Predicted Label: {label_map[pred]}")

if __name__ == "__main__":
    main()
