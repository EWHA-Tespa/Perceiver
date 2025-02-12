import os
import glob
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

import torch
from torch import nn, einsum
from torch.utils.data import Dataset
import torch.nn.functional as F
import models.layers as nl

def tokenize_data(df, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'), MAX_LENGTH=128):
    input_ids, attention_masks = [], []
    
    df['Caption'] = df['Caption'].astype(str).fillna("")

    for text in df['Caption']:
        encoded = tokenizer(
            text, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        input_ids.append(encoded['input_ids'].squeeze(0))
        attention_masks.append(encoded['attention_mask'].squeeze(0))
    return torch.stack(input_ids), torch.stack(attention_masks)

def crop(img_tensor, crop_size=32):
    C, H, W = img_tensor.shape
    img_tensor = img_tensor[:, crop_size:H-crop_size, crop_size:W-crop_size]
    return img_tensor

def patchify(img_tensor, patch_size=16):
    """
    img_tensor: (C, H, W) 형태 (예: (3, 224, 224))
    patch_size: 패치 크기 (16, 16)
    return: (num_patches, patch_dim)
            예) (196, 768)  # (H/16)*(W/16)=14*14=196, 768=16*16*3
    """
    C, H, W = img_tensor.shape
    assert H % patch_size == 0 and W % patch_size == 0, "이미지 크기는 patch_size로 나누어 떨어져야 함"

    # unfold로 (patch_size, patch_size)씩 잘라내기
    patches = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
 
    patches = patches.permute(1, 3, 0, 2, 4).contiguous()

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w

    patches = patches.view(num_patches, -1) 
    return patches

def get_patch_coords(num_patches_h, num_patches_w):

    y_coord = torch.linspace(0, 1, steps=num_patches_h)
    x_coord = torch.linspace(0, 1, steps=num_patches_w)
    grid_y, grid_x = torch.meshgrid(y_coord, x_coord, indexing='ij')  # (14,14) each

    coords = torch.stack([grid_x, grid_y], dim=-1)  # (14,14,2)
    coords = coords.view(-1, 2)                     # (196, 2)
    return coords

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, crop_size=32, patch_size=16, selected_classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.label_encoder = LabelEncoder()
        self.crop_size = crop_size
        self.patch_size = patch_size

        # 데이터 및 레이블 추출
        labels = []
        
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
                if len(image_files) == 1:
                    image_path = image_files[0]
                else:
                    raise ValueError(f"폴더 {folder}에 JPG 파일이 하나가 아닙니다.")

                label_path = os.path.join(folder_path, "label.txt")
                # if os.path.exists(label_path):
                #     with open(label_path, "r") as f:
                #         label = f.read().strip()
                #         labels.append(label)
                #         self.data.append((image_path, label))
                # else:
                #     raise FileNotFoundError(f"폴더 {folder}에 label.txt가 없습니다.")
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"폴더 {folder}에 label.txt가 없습니다.")
                with open(label_path, "r") as f:
                    label = f.read().strip()
                
                if selected_classes is not None and label not in selected_classes:
                    continue
                labels.append(label)
                self.data.append((image_path, label))


        self.label_encoder.fit(labels)
        self.data = [(image_path, self.label_encoder.transform([label])[0]) 
                     for image_path, label in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)  # (3, 224, 224)
        image = crop(image, crop_size=self.crop_size)
        patches = patchify(image, patch_size=self.patch_size)  # (num_patches, patch_dim)

        H, W = image.shape[1], image.shape[2]  
        num_patches_h = H // self.patch_size   
        num_patches_w = W // self.patch_size   
        coords = get_patch_coords(num_patches_h, num_patches_w)  

        combined = torch.cat([patches, coords], dim=1)

        label = torch.tensor(label, dtype=torch.long)
        return combined, label
    
class PerceiverBlock(nn.Module):
    """
    - Cross Attention (latents -> x)
    - 이어서 Self Attention (latent들 끼리)
    - 보통은 LayerNorm, MLP(FeedForward) 등을 곁들여 residual branch를 구성
    """
    def __init__(self, latent_dim, n_heads=8, self_attn_layers=1):
        super().__init__()
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=n_heads)
        self.cross_ln = nn.LayerNorm(latent_dim)  # 잊지 말고 layernorm

        # Self Attention 여러 층
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=n_heads)
            for _ in range(self_attn_layers)
        ])

    def forward(self, latents, x):
        # latents, x: (T, B, dim) 형태로 가정 (주의!)
        # Perceiver 원리상 latents는 query, x는 key/value

        # 1) Cross Attention
        updated_latents, _ = self.cross_attn(latents, x, x)
        latents = latents + updated_latents        # Residual
        latents = self.cross_ln(latents)           # LayerNorm

        # 2) Self Attention 반복
        for layer in self.self_attn_layers:
            latents = layer(latents)  # 내부적으로 residual/LayerNorm 포함

        return latents

class Perceiver(nn.Module):
    def __init__(self, input_dim, latent_dim, latent_size, num_classes,
                 num_blocks, self_attn_layers_per_block=1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_size, latent_dim))
        self.input_projection = nn.Linear(input_dim, latent_dim)

        # 반복될 PerceiverBlock을 여러 개 쌓는다.
        self.blocks = nn.ModuleList([
            PerceiverBlock(
                latent_dim=latent_dim,
                n_heads=8,
                self_attn_layers=self_attn_layers_per_block
            )
            for _ in range(num_blocks)
        ])

        # Feature Extractor를 self.shared로 설정
        self.shared = nn.ModuleList([self.input_projection] + list(self.blocks))

        self.output_layer = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        """
        x: (B, T, F) = (배치, 시퀀스길이, 피처차원)
        """
        B, T, F = x.size()
        x = self.input_projection(x)                 # (B, T, latent_dim)

        # latents: (latent_size, latent_dim) -> 배치 차원 확장 (B, latent_size, latent_dim)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        # MultiHeadAttention은 (seq, batch, dim) 순서를 권장하므로 permute
        x = x.permute(1, 0, 2)        # (T, B, latent_dim)
        latents = latents.permute(1, 0, 2)  # (latent_size, B, latent_dim)

        # 여러 개의 PerceiverBlock 반복
        for block in self.blocks:
            latents = block(latents, x)

        # 최종 latents: (latent_size, B, latent_dim)
        latents = latents.permute(1, 0, 2).mean(dim=1)  # (B, latent_dim)
        return self.output_layer(latents)

class CombinedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, perceiver_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.perceiver = perceiver_model

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)  # (B, T, embed_dim)
        return self.perceiver(embeddings)  
