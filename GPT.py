import torch
import torch.nn as nn

from MoziCore.transformer.Transformer import DecoderTransformerBlock, EncoderTransformerBlock
from MoziCore.normalizer.LayerNorm import LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DecoderTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

        self.pos_emb_encoder = nn.Embedding(cfg["encode_context_length"], cfg["emb_dim"])
        self.drop_emb_encoder = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks_encoder = nn.Sequential(
            *[EncoderTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        elf.final_norm_encoder = LayerNorm(cfg["emb_dim"])

    def forward(self, in_idx, input_eng):
        
        b, eng_len, eng_d_in = input_eng.shape
        pos_emdebs_eng = self.pos_emb(torch.arange(eng_len, device=input_eng.device))
        xEng = input_eng + pos_emdebs_eng
        xEng = self.drop_emb_encoder(xEng)
        xEng = self.trf_blocks_encoder(xEng)
        xEng = self.final_norm_encoder(xEng)

        batch_size, seq_len, d_in = in_idx.shape
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = in_idx + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
