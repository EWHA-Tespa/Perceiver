import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertTokenizer
import utils.manager as Manager 
from models.shared_perceiver import Perceiver, CombinedModel

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# 경로 설정
pruned_model_dir = "/home/jisoo/Perceiver/checkpoints_pruned3"
loader_dir = "/home/jisoo/Perceiver/loader/"
batch_size = 32

# Pruned 모델 로드 함수
def load_pruned_models(pruned_model_dir):
    pruned_text_models, pruned_image_models = [], []
    
    for i in range(6):
        file_path = f"{pruned_model_dir}/text_checkpoint_{i+1}_epoch_40.pth.tar"
        model = torch.load(file_path, map_location=device)
        pruned_text_models.append(model)

    for i in range(6):
        file_path = f"{pruned_model_dir}/image_checkpoint_{i+1}_epoch_40.pth.tar"
        model = torch.load(file_path, map_location=device)
        pruned_image_models.append(model)

    return pruned_text_models, pruned_image_models

# Validation 데이터 로드 함수
def load_valid_loaders():
    valid_loaders = []
    for i in range(6):
        with open(f"{loader_dir}text_val_loader_{i+1}.pkl", 'rb') as f:
            valid_loaders.append(pickle.load(f))
    for i in range(6):
        with open(f"{loader_dir}image_val_loader_{i+1}.pkl", 'rb') as f:
            valid_dataset = pickle.load(f)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            valid_loaders.append(valid_loader)
    return valid_loaders

# Knowledge Transfer 수행 함수
def knowledge_transfer(task_id, backbone_id, backbone_modality, task_modality, 
                       task_models, backbone_models, valid_loaders, criterion, device, epochs=20):
    backbone_model = backbone_models[backbone_id] if backbone_modality == "Text" else backbone_models[backbone_id - 6]
    task_model = task_models[task_id]
    task_model.to(device)

    optimizer = optim.SGD(task_model.parameters(), lr=0.001, momentum=0.9)
    task_model.train()
    shared_layer_info = {}

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch in valid_loaders[task_id]:
            optimizer.zero_grad()
            if task_modality == "Text":
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = task_model(inputs)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = task_model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Task {task_id} | Backbone {backbone_id} | Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Accuracy: {correct/total:.4f}")
    
    # Checkpoint 저장
    Manager.save_checkpoint(task_model, optimizer, epoch + 1, pruned_model_dir, shared_layer_info, dataset=f"task_{task_id}")
    print(f"[Task {task_id}] Knowledge Transfer 완료 및 체크포인트 저장")
    return task_model

# 실행 코드
if __name__ == "__main__":
    print("Pruned 모델 및 Validation 로더 로드 중...")
    pruned_text_models, pruned_image_models = load_pruned_models(pruned_model_dir)
    valid_loaders = load_valid_loaders()
    pruned_models = pruned_text_models + pruned_image_models

    print("Cosine-based Knowledge Transfer 시작...")
    for task_id in range(6):
        knowledge_transfer(task_id, task_id, "Text", "Text", pruned_models, pruned_models, valid_loaders, criterion, device)
    print("Knowledge Transfer 완료 및 체크포인트 저장됨.")
