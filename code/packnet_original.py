import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader
import pickle
import random
import numpy as np
import pandas as pd
import os

def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정

def seed_worker(worker_id): #데이터로더 난수고정
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed_everything(42)
g = torch.Generator()
g.manual_seed(42)
NUM_WORKERS = 4 # 서브프로세스관리자 수. 난수생성과 관련있습니다. 일단은 4로 고정합니다.

# Perceiver 모델 가져오기
import sys
sys.path.append("/home/jisoo/Perceiver/code")
from perceiver import crop, patchify, get_patch_coords, ImageDataset, PerceiverBlock, Perceiver, CustomDataset, CombinedModel
torch.serialization.add_safe_globals([CombinedModel])

# SparsePruner 가져오기
sys.path.append("/home/jisoo/Perceiver/code")
from sparse_pruner_packnet import SparsePruner

root_dir = '/home/jisoo/Perceiver/model/'
loader_dir = '/home/jisoo/Perceiver/loader/'
batch_size = 32

input_models = OrderedDict()
valid_loaders = OrderedDict()

for i in range(6):
    model_path = root_dir + f"text_model_{i+1}.pkl"
    try:
        input_models[f"text_model_{i+1}"] = torch.load(model_path, map_location="cpu", weights_only=False)
        print(f"텍스트 모델 {i+1} 불러오기 완료: {model_path}")
    except AttributeError as e:
        print(f"*오류 발생: {e}")
        print(f"*`weights_only=True`로 다시 로드 시도")
        model_weights = torch.load(model_path, map_location="cpu", weights_only=True)

        # Perceiver 모델을 새로 생성하고 가중치 적용
        model = Perceiver(input_dim=768, latent_dim=512, latent_size=128, num_classes=100, num_blocks=6, self_attn_layers_per_block=2)
        model.load_state_dict(model_weights)

        input_models[f"text_model_{i+1}"] = model
        print(f"텍스트 모델 {i+1} (weights-only) 불러오기 완료: {model_path}")


for i in range(6):
    model_path = root_dir + f"image_model_{i+1}.pkl"
    input_models[f"image_model_{i+1}"] = torch.load(model_path, map_location="cpu", weights_only=False)
    print(f"image model {i+1} 불러오기 완료: {model_path}")

for i in range(6):
    loader_path = loader_dir + f"text_val_loader_{i+1}.pkl"
    with open(loader_path, "rb") as f:
        valid_loaders[f"text_loader_{i+1}"] = pickle.load(f)
    print(f"text data loader {i+1} 불러오기 완료: {loader_path}")

for i in range(6):
    loader_path = loader_dir + f"image_val_loader_{i+1}.pkl"
    with open(loader_path, "rb") as f:
        loaded_valid_dataset = pickle.load(f)

    valid_loaders[f"image_loader_{i+1}"] = DataLoader(loaded_valid_dataset, batch_size=batch_size, shuffle=False)
    print(f"image data loader {i+1} 불러오기 완료: {loader_path}")


class Manager:
    """Handles pruning, fine-tuning, and inference for Perceiver models."""

    def __init__(self, args, model, previous_masks):
        self.args = args
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # 이전 태스크의 마스크 불러오기
        self.previous_masks = previous_masks
        self.criterion = nn.CrossEntropyLoss()

        # PackNet 방식으로 Pruning을 위한 SparsePruner 설정
        self.pruner = SparsePruner(
            model=self.model,
            prune_perc=self.args.prune_perc,
            previous_masks=self.previous_masks,
            train_bias=False,
            train_bn=False
        )

    def prune(self):
        """Perform pruning on the model."""
        print("** Pruning 시작...")
        self.pruner.prune()
        print("** Pruning 후 모델 성능 평가...")
        self.eval()

    def train(self, train_loader, optimizer, epochs):
        """Train the model with fine-tuning."""
        print("** Finetuning을 위한 모델 학습 시작...")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                inputs = batch["input_ids"].to(self.device).float()
                attention_mask = batch["attention_mask"].to(self.device).float()
                labels = batch["labels"].to(self.device)

                inputs = inputs.reshape(inputs.shape[0], -1)
                attention_mask = attention_mask.reshape(attention_mask.shape[0], -1)

                # 백본 모델이 `image_model_`이고, 데이터 로더가 `text_loader_`일 때만
                if "image_model" in self.args.backbone_model and "text_loader" in self.args.data_loader:
                    inputs = torch.nn.functional.pad(inputs, (0, 770 - 128))  # (B, 128) → (B, 770)

                if inputs.dim() == 2: # 입력 차원 (B, F) -> (B, 1, F)
                    inputs = inputs.unsqueeze(1)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # PackNet: pruning된 가중치는 업데이트되지 않도록 처리
                self.pruner.make_grads_zero()
                optimizer.step()

                # pruning된 가중치는 항상 0으로 유지
                self.pruner.make_pruned_zero()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}: Loss={total_loss:.4f}")

        print("Finetuning 완료!")

    def eval(self):
        """Evaluate the model after pruning and fine-tuning."""
        print("** 모델 평가 시작...")
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.args.test_loader:
                inputs = batch["input_ids"].to(self.device).float()
                attention_mask = batch["attention_mask"].to(self.device).float()
                labels = batch["labels"].to(self.device) 

                inputs = inputs.reshape(inputs.shape[0], -1)
                attention_mask = attention_mask.reshape(attention_mask.shape[0], -1)

                self.pruner.apply_mask(self.pruner.current_dataset_idx)

                if inputs.dim() == 2: # 입력 차원 (B, F) -> (B, 1, F)
                    inputs = inputs.unsqueeze(1)
                
                # 백본 모델이 `image_model_`이고, 데이터 로더가 `text_loader_`일 때만
                if "image_model" in self.args.backbone_model and "text_loader" in self.args.data_loader:
                    inputs = torch.nn.functional.pad(inputs, (0, 770 - 128))  # (B, 128) → (B, 770)


                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"모델 평가 완료 - Accuracy: {accuracy:.2f}%")
        return accuracy

    def save_model(self, save_path):
        """Save the model after training."""
        print("** 모델 저장 중...")
        torch.save(self.model.state_dict(), save_path)
        print(f"모델 저장 완료: {save_path}")


#  데이터 로드 함수 (Pickle 파일에서 `DataLoader` 불러오기)
def load_dataloader(pkl_path):
    print(f"{pkl_path}에서 DataLoader 로드 중...")
    with open(pkl_path, "rb") as f:
        dataloader = pickle.load(f)
    return dataloader


# 실행 코드
def packnet_train_prune_infer(args, save_path):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # 1. 백본 모델 로드 (`image_model_2.pkl`)
    # print("백본 모델 로드 중...")
    # model = CombinedModel(vocab_size=30522, embed_dim=768, perceiver_model=Perceiver(
    #     input_dim=768,       # BERT 기반 텍스트 입력 차원 또는 이미지 패치 차원
    #     latent_dim=512,      # Perceiver의 Latent 차원
    #     latent_size=128,     # Latent 개수
    #     num_classes=100,     # Perceiver 모델 내에서 num_classes를 지정
    #     num_blocks=6,        # Perceiver 블록 개수
    #     self_attn_layers_per_block=2   # Self-Attention 레이어 개수
    # )).to(device)
    # model_checkpoint = torch.load("/home/jisoo/Perceiver/model/image_model_2.pkl", map_location=device, weights_only=True)

    # model.load_state_dict(model_checkpoint)

    # 1. 백본 모델 로드 (`image_model_2.pkl`)
    print("** 백본 모델 로드 중...")
    model = input_models["image_model_2"].to(device)

    # 2. PackNet 방식으로 백본 모델에서 Pruning 수행
    print("** 백본 모델에서 Pruning 수행...")
    previous_masks = OrderedDict()
    for module_idx, module in enumerate(model.modules()):
        if isinstance(module, nn.Linear):
            previous_masks[module_idx] = torch.ones_like(module.weight.data).to(device)

    manager = Manager(args, model, previous_masks)
    manager.prune()

    print("** Fine-tuning을 위한 Pruning Mask 적용...")
    manager.pruner.make_finetuning_mask()

    # 3. Text 데이터로 Finetuning (`text_val_loader_1.pkl`)
    print("** `valid_loaders`에서 데이터 로더 가져오기...")
    text_loader = valid_loaders["text_loader_1"]

    print("** Text 데이터로 Finetuning 시작...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    manager.train(text_loader, optimizer, args.epochs)

    # 4. Inference 수행
    print("** Inference 검증 시작...")
    accuracy = manager.eval()
    print(f"최종 Accuracy: {accuracy:.2f}%")

    # 5. 최종 모델 저장
    manager.save_model(save_path)

class Args:
    cuda = True
    num_classes = 100
    prune_perc = 0.1
    lr = 1e-4
    epochs = 20
    backbone_model = "image_model_2" 
    data_loader = "text_loader_1" 
    test_loader = load_dataloader("/home/jisoo/Perceiver/loader/text_val_loader_1.pkl")  # Inference를 위한 데이터 로드

args = Args()
packnet_train_prune_infer(
    args,
    save_path="/home/jisoo/Perceiver/model/packnet_finetuned_model.pth"
)