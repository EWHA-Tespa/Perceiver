class SparsePruner:
    """Performs pruning on the given model (without self.shared)."""

    def __init__(self, model, prune_perc, previous_masks):
        self.model = model
        self.prune_perc = prune_perc
        self.current_masks = {}
        self.previous_masks = previous_masks or {}

    def pruning_mask(self, weights, previous_mask):
        """Ranks weights by magnitude and sets all below kth percentile to 0."""
        previous_mask = previous_mask.cuda()
        tensor = weights[previous_mask.eq(1)]  # 현재 학습 가능한 가중치만 고려
        if tensor.numel() == 0:
            return previous_mask

        abs_tensor = tensor.abs().view(-1).cpu()
        cutoff_rank = round(self.prune_perc * tensor.numel())

        if cutoff_rank < 1:
            return previous_mask

        cutoff_value = abs_tensor.kthvalue(cutoff_rank)[0].item()
        remove_mask = (weights.abs() <= cutoff_value) & previous_mask.eq(1)
        previous_mask[remove_mask] = 0  # Pruned 가중치는 0으로 설정
        return previous_mask

    def prune(self):
        """Computes pruning masks and applies them to the model (without self.shared)."""
        print(f"Pruning {self.prune_perc * 100:.2f}% of the weights")

        for module_idx, module in enumerate(self.model.modules()):  # self.shared 대신 self.model 사용
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                previous_mask = self.previous_masks.get(module_idx, torch.ones_like(weight, device=weight.device))
                mask = self.pruning_mask(weight, previous_mask)

                # 업데이트된 마스크 저장
                self.current_masks[module_idx] = mask.cuda()
                weight[mask.eq(0)] = 0  # Pruned된 가중치를 0으로 설정

        return self.current_masks
