import sys
import os
import torch
import torch.nn as nn
import pickle
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
from transformers import BertTokenizer

shared_perceiver_path = "/home/jisoo/Perceiver/code/models"
sys.path.append(shared_perceiver_path) 
from models.shared_perceiver import Perceiver, CombinedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

seed_everything(42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LENGTH = 128

root_dir = "/home/jisoo/Perceiver/shared_layer_model/"
loader_dir = "/home/jisoo/Perceiver/loader/"

batch_size = 32

input_models = []
valid_loaders = []

for i in range(6):
    model_path = os.path.join(root_dir, f"text_checkpoint_{i+1}_epoch_40.pth.tar")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않음: {model_path}")
    text_model = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    input_models.append(text_model)
    print(f"Text model {i+1}번 불러오기 완료: {model_path}")

for i in range(6): 
    model_path = os.path.join(root_dir, f"image_checkpoint_{i}_epoch_40.pth.tar")
    img_model = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    input_models.append(img_model)
    print(f"Image model {i}번 불러오기 완료: {model_path}")

for i in range(6):
    loader_path = os.path.join(loader_dir, f"text_val_loader_{i+1}.pkl")
    with open(loader_path, 'rb') as f:
        loaded_valid_dataset = pickle.load(f)
    valid_loaders.append(loaded_valid_dataset)
    print(f"Text val. loader {i+1}번 불러오기 완료.")

for i in range(6):
    loader_path = os.path.join(loader_dir, f"image_val_loader_{i+1}.pkl")
    with open(loader_path, 'rb') as f:
        loaded_valid_dataset = pickle.load(f)
    valid_loader = DataLoader(loaded_valid_dataset, batch_size=batch_size, shuffle=False)
    valid_loaders.append(valid_loader)
    print(f"Image val. loader {i+1}번 불러오기 완료.")

class PackNet(nn.Module):
    def __init__(self, model):
        super(PackNet, self).__init__()
        self.model = model
        self.masks = {}
        self.current_task = None

    def set_task(self, task_id):
        self.current_task = task_id
        if task_id not in self.masks:
            self.masks[task_id] = {
                name: torch.ones_like(param, device=param.device, dtype=torch.float32)
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }

    def apply_mask(self):
        if self.current_task not in self.masks:
            return 
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    mask = self.masks[self.current_task][name]
                    param.mul_(mask)

    def prune(self, prune_perc=0.2):
        if self.current_task is None:
            raise ValueError("Task must be set before pruning.")

        current_mask_dict = self.masks[self.current_task]

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                mask = current_mask_dict[name]
                param_data = param.data[mask.eq(1)]
                if param_data.numel() == 0:
                    continue

                abs_tensor = param_data.abs().view(-1).cpu()
                cutoff_rank = int(round(prune_perc * param_data.numel()))
                
                if cutoff_rank < 1:
                    continue

                cutoff_value = abs_tensor.kthvalue(cutoff_rank)[0].item()
                to_zero = (param.abs() <= cutoff_value) & (mask.eq(1))
                mask[to_zero] = 0.0

                current_mask_dict[name] = mask

        self.apply_mask()

    def forward(self, input_ids, **kwargs):
        self.apply_mask()
        return self.model(input_ids, **kwargs)

def eval_epoch(model, dataloader, criterion, device, is_text: bool):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if is_text:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def gradual_pruning(packnet_model, model_type, model_index, criterion, device, 
                    start_sparsity, end_sparsity, pruning_steps, checkpoint_dir, valid_loaders):

    model_path = f"{checkpoint_dir}/{model_type}_model_{model_index+1}_pruned.pkl"
    sparsity_increment = (end_sparsity - start_sparsity) / pruning_steps
    current_sparsity = start_sparsity

    if model_type == "text":
        test_loader = valid_loaders[model_index]
        is_text = True
    else:
        test_loader = valid_loaders[model_index + 6]
        is_text = False

    for step in range(pruning_steps):
        print(f"[{model_type.upper()} Model {model_index+1}] Pruning Step {step+1}/{pruning_steps}, "
              f"Target prune perc={current_sparsity:.2f}")
        packnet_model.prune(prune_perc=current_sparsity)
        current_sparsity += sparsity_increment

    with open(model_path, 'wb') as f:
        pickle.dump({
            "model_state_dict": packnet_model.state_dict(),
            "masks": packnet_model.masks
        }, f)
    print(f"[{model_type.upper()} Model {model_index+1}] Pruned model saved: {model_path}")

    test_loss, test_acc = eval_epoch(packnet_model, test_loader, criterion, device, is_text=is_text)
    print(f"[{model_type.upper()} Model {model_index+1}] Final Test Accuracy: {test_acc:.4f}")
    print("---------")
    
    return packnet_model

if __name__ == "__main__":
    start_sparsity = 0.05
    end_sparsity = 0.2
    pruning_steps = 5
    checkpoint_dir = "/home/youlee/perceiver/perceiver/checkpoints_pruned3"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    print("Starting gradual pruning process...")

    text_models = input_models[:6]
    image_models = input_models[6:]

    pruned_text_models = []
    for i, model in enumerate(text_models):
        packnet_model = PackNet(model)
        packnet_model.set_task(f"text_task_{i+1}")
        packnet_model.to(device)

        pruned_text_models.append(gradual_pruning(packnet_model, "text", i, criterion, device, start_sparsity, end_sparsity, pruning_steps, checkpoint_dir, valid_loaders))

    pruned_image_models = []
    for i, model in enumerate(image_models):
        packnet_model = PackNet(model)
        packnet_model.set_task(f"image_task_{i+1}")
        packnet_model.to(device)

        pruned_image_models.append(gradual_pruning(packnet_model, "image", i, criterion, device, start_sparsity, end_sparsity, pruning_steps, checkpoint_dir, valid_loaders))

    print("Gradual pruning process finished.")