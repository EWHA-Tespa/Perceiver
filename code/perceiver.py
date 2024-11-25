import numpy as np
import pandas as pd
import torch
from torch import nn, einsum
import torch.nn.functional as F
#from perceiver_pytorch.perceiver_pytorch import exists, default, cache_fn, fourier_encode, PreNorm, FeeodForward, Attention

class CrossAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super(CrossAttention, self).__init__()
        self.key_proj = nn.Linear(d_in, d_out_kq)
        self.query_proj = nn.Linear(d_in, d_out_kq)
        self.value_proj = nn.Linear(d_in, d_out_v)
        self.softmax = nn.Softmax(dim=-1)           # 이게 뭐지

    def forward(self, x, latent):
        keys = self.key_proj(x)
        queries = self.query_proj(latent)
        values = self.value_proj(x)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_probs = self.softmax(attention_scores)

        attended_values = torch.matmul(attention_probs, values)
        return attended_values
    
class LatentTransformer(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, embed_dim):
        super(LatentTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads) # trasformer 로 latent array 반복적으로 update
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, latent):
        latent = latent.permute(1,0,2)  # Transformer는 (seq_len, batch_size, latent_dim) 형식으로 데이터 받음.
        latent = self.transformer(latent)
        return latent.permute(1,0,2)    # 이걸 다시 (batch_size, latent_len, latent_dim으로 바꿈)
    
class Averaging(nn.Module):
    def forward(self, latent):
        return latent.mean(dim=1)   # latent vector를 평균내서 최종 logits 계산
    
class Perceiver(nn.Module):
    def __init__(self, input_dim, latent_dim, embed_dim, num_heads, num_layers, num_classes):
        super(Perceiver, self).__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.latents = nn.Parameter(torch.randn(1, latent_dim, embed_dim))

        self.cross_attention = CrossAttention(d_in=embed_dim, d_out_kq=embed_dim, d_out_v=embed_dim)
        self.latent_transformer = LatentTransformer(latent_dim=latent_dim, num_heads=num_heads, 
                                                    num_layers=num_layers, embed_dim=embed_dim)
        
        self.averaging = Averaging()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)
        latent = self.latents.repeat(batch_size, 1, 1)    # batch 학습 시, batch 내 각각 샘플에 서로 다른 독립적인 latent값을 제공
        latent = self.cross_attention(x, latent)
        latent = self.latent_transformer(latent)
        latent_avg = self.averaging(latent)
        logits = self.classifier(latent_avg)
        return logits