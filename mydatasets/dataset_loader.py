import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BartTokenizerFast

class CNNDMDataset(Dataset):
    def __init__(self, split, tokenizer, max_input_length=512, max_sentences=32):
        self.raw = load_dataset("cnn_dailymail", "3.0.0", split=split)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        article = self.raw[idx]["article"]
        summary = self.raw[idx]["highlights"]
        # tokenize entire article
        enc = self.tokenizer(article, truncation=True, padding="max_length", 
                             max_length=self.max_input_length, return_tensors="pt")
        # split into sentences
        sentences = article.split(". ")
        sentences = sentences[:self.max_sentences]
        if len(sentences) < self.max_sentences:
            sentences += [""] * (self.max_sentences - len(sentences))

        sent_enc = self.tokenizer(sentences, truncation=True, padding="max_length", 
                                   max_length=self.max_input_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "sent_input_ids": sent_enc["input_ids"],
            "sent_attention_mask": sent_enc["attention_mask"],
            # labels can be built later for extractive head
        }

def get_dataloader(split, tokenizer, batch_size, shuffle=False):
    dataset = CNNDMDataset(split, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)