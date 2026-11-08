# Appendix E: Parameter-efficient fine-tuning with LoRA
# Same spam task as Chapter 6, but instead of updating the original weights W,
# we freeze them and learn a small low-rank update  W + alpha * (A @ B).

import math
import time

import tiktoken
import torch
import torch.nn as nn

from ch05_pretraining import load_pretrained_gpt2
from ch06_classification_finetuning import (calc_accuracy_loader, create_spam_loaders,
                                            train_classifier_simple)
from previous_chapters import get_device


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # same init as nn.Linear
        self.B = nn.Parameter(torch.zeros(rank, out_dim))  # zeros -> starts as "no change"
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear  # original (frozen) layer
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    # Recursively swap every nn.Linear for a LinearWithLoRA
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()
    train_loader, val_loader, test_loader, max_length = create_spam_loaders(tokenizer)

    model, cfg = load_pretrained_gpt2("gpt2-small (124M)")
    torch.manual_seed(123)
    model.out_head = nn.Linear(cfg["emb_dim"], 2)

    # Freeze ALL original weights (including the new head)...
    print(f"Trainable params before: {count_trainable(model):,}")  # ~124M
    for param in model.parameters():
        param.requires_grad = False
    print(f"Trainable params after freezing: {count_trainable(model):,}")  # 0

    # ...then add LoRA adapters: only these are trained
    replace_linear_with_lora(model, rank=16, alpha=16)
    print(f"Trainable LoRA params: {count_trainable(model):,}")  # ~2.67M (~50x fewer)
    model.to(device)

    torch.manual_seed(123)
    print(f"Validation accuracy (before): {calc_accuracy_loader(val_loader, model, device, 10) * 100:.2f}%")

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs=5, eval_freq=50, eval_iter=5)
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

    print(f"Training accuracy: {calc_accuracy_loader(train_loader, model, device) * 100:.2f}%")
    print(f"Validation accuracy: {calc_accuracy_loader(val_loader, model, device) * 100:.2f}%")
    print(f"Test accuracy: {calc_accuracy_loader(test_loader, model, device) * 100:.2f}%")
