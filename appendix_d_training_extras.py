# Appendix D: Adding bells and whistles to the training loop
#   1) learning rate warmup  2) cosine decay  3) gradient clipping

import math

import tiktoken
import torch

from ch04_gpt_model import GPTModel
from ch05_pretraining import (GPT_CONFIG_124M, calc_loss_batch, evaluate_model,
                              generate_and_print_sample, load_the_verdict)
from previous_chapters import create_dataloader_v1, get_device


def train_model(model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq,
                eval_iter, start_context, tokenizer, warmup_steps, initial_lr=3e-05, min_lr=1e-6):
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    peak_lr = optimizer.param_groups[0]["lr"]
    total_training_steps = len(train_loader) * n_epochs
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # 1) Linear warmup, then 2) cosine decay to min_lr
            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment
            else:
                progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # 3) Gradient clipping after warmup (caps the gradient norm at 1.0)
            if global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader,
                                                      device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen, track_lrs


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()
    text_data = load_the_verdict()
    split_idx = int(0.90 * len(text_data))

    torch.manual_seed(123)
    ctx = GPT_CONFIG_124M["context_length"]
    train_loader = create_dataloader_v1(text_data[:split_idx], batch_size=2, max_length=ctx,
                                        stride=ctx, drop_last=True, shuffle=True)
    val_loader = create_dataloader_v1(text_data[split_idx:], batch_size=2, max_length=ctx,
                                      stride=ctx, drop_last=False, shuffle=False)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M).to(device)
    peak_lr = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)

    n_epochs = 15
    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(0.2 * total_steps)  # warm up for 20% of training

    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, train_loader, val_loader, optimizer, device, n_epochs=n_epochs,
        eval_freq=5, eval_iter=1, start_context="Every effort moves you",
        tokenizer=tokenizer, warmup_steps=warmup_steps, initial_lr=1e-5, min_lr=1e-5,
    )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 3))
    plt.plot(range(len(lrs)), lrs)
    plt.xlabel("Steps")
    plt.ylabel("Learning rate")
    plt.title("Warmup + cosine decay")
    plt.tight_layout()
    plt.show()
