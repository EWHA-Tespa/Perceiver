import torch
import torch.nn as nn
import pdb

import models.layers as nl


def save_checkpoint(model, optimizer, epoch_idx, save_folder, shared_layer_info, dataset, idx=None):
    """모델의 체크포인트를 저장하는 함수"""
    if idx is not None:
        filepath = f"{save_folder}/{dataset}_checkpoint_{idx}_epoch_{epoch_idx}.pth.tar"
    else:
        filepath = f"{save_folder}/{dataset}_checkpoint_epoch_{epoch_idx}.pth.tar"
    if dataset not in shared_layer_info:
        shared_layer_info[dataset] = {
            'bias': {},
            'piggymask': {},
            'bn_layer_running_mean': {},
            'bn_layer_running_var': {},
            'bn_layer_weight': {},
            'bn_layer_bias': {},
            'prelu_layer_weight': {}
        }
    # 모델 내부 `shared_layer_info` 업데이트
    for name, module in model.named_modules():
        if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
            if module.bias is not None:
                shared_layer_info[dataset]['bias'][name] = module.bias
            if module.piggymask is not None:
                shared_layer_info[dataset]['piggymask'][name] = module.piggymask
        elif isinstance(module, nn.BatchNorm2d):
            shared_layer_info[dataset]['bn_layer_running_mean'][name] = module.running_mean
            shared_layer_info[dataset]['bn_layer_running_var'][name] = module.running_var
            shared_layer_info[dataset]['bn_layer_weight'][name] = module.weight
            shared_layer_info[dataset]['bn_layer_bias'][name] = module.bias
        elif isinstance(module, nn.PReLU):
            shared_layer_info[dataset]['prelu_layer_weight'][name] = module.weight

    # 체크포인트 저장할 딕셔너리
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'dataset_history': model.perceiver.datasets,
        'dataset2num_classes': model.perceiver.dataset2num_classes,
        'masks': model.perceiver.masks if hasattr(model.perceiver, "masks") else None,
        'shared_layer_info': shared_layer_info,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch_idx
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath}")


def load_checkpoint(model, optimizers, resume_from_epoch, save_folder):

    if resume_from_epoch > 0:
        filepath = save_folder
        checkpoint = torch.load(filepath)
        checkpoint_keys = checkpoint.keys()
        state_dict = checkpoint['model_state_dict']
        curr_model_state_dict = model.state_dict() #.module.state_dict()

        for name, param in state_dict.items():
            if ('piggymask' in name or name == 'classifier.weight' or name == 'classifier.bias' or
                (name == 'classifier.0.weight' or name == 'classifier.0.bias' or name == 'classifier.1.weight')):
                # I DONT WANT TO DO THIS! QQ That last 3 exprs are for anglelinear and embeddings
                continue
            elif len(curr_model_state_dict[name].size()) == 4:
                # Conv layer
                curr_model_state_dict[name][:param.size(0), :param.size(1), :, :].copy_(param)
            elif len(curr_model_state_dict[name].size()) == 2 and 'features' in name:
                # FC conv (feature layer)
                curr_model_state_dict[name][:param.size(0), :param.size(1)].copy_(param)                
            elif len(curr_model_state_dict[name].size()) == 1:
                # bn and prelu layer
                curr_model_state_dict[name][:param.size(0)].copy_(param)
            elif 'classifiers' in name:
                curr_model_state_dict[name][:param.size(0), :param.size(1)].copy_(param)
            else:
                try:
                    curr_model_state_dict[name].copy_(param)
                except:
                    pdb.set_trace()
                    print("There is some corner case that we haven't tackled")
    return