import torch
import torch.nn as nn
from transformers import BartModel

class HybridSummarizer(nn.Module):
    def __init__(self, abstractive_model_name, extractor_hidden, extractor_layers):
        super().__init__()
        # Abstractive encoder
        self.abstractive = BartModel.from_pretrained(abstractive_model_name)
        d_model = self.abstractive.config.hidden_size
        # Simple Transformer encoder for sentence representations
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.extractor = nn.TransformerEncoder(encoder_layer, num_layers=extractor_layers)
        # Cross-attention: query = sentence embeddings, key/value = token embeddings
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        # Classification head
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, input_ids, attention_mask, sent_input_ids, sent_attention_mask):
        # 1. get token embeddings from abstractive encoder
        outputs = self.abstractive(input_ids=input_ids, attention_mask=attention_mask)
        token_feats = outputs.last_hidden_state  # [B, L, D]
        # 2. encode sentences via abstractive encoder & average pool
        bs, ns, sl = sent_input_ids.size()
        sent_embs = []
        for i in range(bs):
            se = sent_input_ids[i]  # [ns, sl]
            sm = sent_attention_mask[i]
            out = self.abstractive(input_ids=se, attention_mask=sm).last_hidden_state  # [ns, sl, D]
            pooled = out.mean(dim=1)  # [ns, D]
            sent_embs.append(pooled)
        sent_feats = torch.stack(sent_embs, dim=0)  # [B, ns, D]
        # 3. cross-attention
        attn_out, _ = self.cross_attn(query=sent_feats, key=token_feats, value=token_feats)
        # 4. transformer extractor layers
        ext_out = self.extractor(attn_out)  # [B, ns, D]
        # 5. classify each sentence
        logits = self.classifier(ext_out)     # [B, ns, 1]
        return logits.squeeze(-1)  # [B, ns]