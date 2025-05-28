import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BartTokenizerFast
from mydatasets.dataset_loader import get_dataloader
from models.hybrid_model import HybridSummarizer
from utils.rouge_utils import compute_rouge

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def train():
    cfg = load_config()
    tokenizer = BartTokenizerFast.from_pretrained(cfg["model"]["abstractive_model"])
    train_loader = get_dataloader("train", tokenizer, cfg["training"]["batch_size"], shuffle=True)
    val_loader   = get_dataloader("validation", tokenizer, cfg["training"]["batch_size"])

    model = HybridSummarizer(
        abstractive_model_name=cfg["model"]["abstractive_model"],
        extractor_hidden=cfg["model"]["extractor_hidden"],
        extractor_layers=cfg["model"]["extractor_layers"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(cfg["training"]["save_dir"], exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            sent_ids   = batch["sent_input_ids"].to(device)
            sent_mask  = batch["sent_attention_mask"].to(device)
            # TODO: build labels for extractive selection
            labels = torch.zeros(input_ids.size(0), sent_ids.size(1), device=device)

            logits = model(input_ids, attn_mask, sent_ids, sent_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            rouge_scores = []
            for batch in val_loader:
                # similar steps, generate extractive summary then compute rouge
                pass
            print(f"Epoch {epoch+1} done. \n")

        # save
        torch.save(model.state_dict(), os.path.join(cfg["training"]["save_dir"], f"model_epoch{epoch+1}.pt"))

if __name__ == '__main__':
    train()