from torch.utils.data import Dataset
from transformers import T5Tokenizer

class PolicyPredictionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = "condition: " + " ".join(item["parsed"]["conditions"])
        target_text = item["parsed"]["conclusion"]

        inputs = self.tokenizer(
            input_text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        targets = self.tokenizer(
            target_text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }
