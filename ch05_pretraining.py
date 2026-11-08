# Chapter 5: Pretraining on unlabeled data

import os
import urllib.request

import numpy as np
import tiktoken
import torch

from ch04_gpt_model import GPTModel, generate_text_simple
from previous_chapters import create_dataloader_v1, get_device

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # shortened from 1024 so training fits on a laptop
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


# ---------------------------------------------------------------
# 5.1 Text <-> token ID helpers
# ---------------------------------------------------------------
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)  # add batch dimension


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


# ---------------------------------------------------------------
# 5.1 Loss functions
# ---------------------------------------------------------------
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # cross_entropy wants (N, classes) and (N,) -> flatten batch and sequence dims
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches


# ---------------------------------------------------------------
# 5.2 Training loop
# ---------------------------------------------------------------
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, max_new_tokens=50,
                                         context_size=context_size)
    print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()             # reset gradients from previous batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                   # compute gradients
            optimizer.step()                  # update weights
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader,
                                                      device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, filename="loss-plot.pdf"):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()                       # second x-axis for tokens seen
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()


# ---------------------------------------------------------------
# 5.3 Better decoding: temperature scaling + top-k sampling
# ---------------------------------------------------------------
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            # keep only the k largest logits, set the rest to -inf
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1:]
            logits = torch.where(logits < min_val,
                                 torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature    # >1 flatter (more random), <1 sharper
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # greedy

        if eos_id is not None and idx_next.item() == eos_id:
            break  # stop at end-of-text token (batch size 1)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ---------------------------------------------------------------
# 5.5 Loading pretrained weights from OpenAI
# ---------------------------------------------------------------
MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        blk = params["blocks"][b]
        att = gpt.trf_blocks[b].att

        # OpenAI stores Q, K, V in one matrix "c_attn" -> split into 3
        q_w, k_w, v_w = np.split(blk["attn"]["c_attn"]["w"], 3, axis=-1)
        att.W_query.weight = assign(att.W_query.weight, q_w.T)
        att.W_key.weight = assign(att.W_key.weight, k_w.T)
        att.W_value.weight = assign(att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(blk["attn"]["c_attn"]["b"], 3, axis=-1)
        att.W_query.bias = assign(att.W_query.bias, q_b)
        att.W_key.bias = assign(att.W_key.bias, k_b)
        att.W_value.bias = assign(att.W_value.bias, v_b)

        att.out_proj.weight = assign(att.out_proj.weight, blk["attn"]["c_proj"]["w"].T)
        att.out_proj.bias = assign(att.out_proj.bias, blk["attn"]["c_proj"]["b"])

        ff = gpt.trf_blocks[b].ff.layers
        ff[0].weight = assign(ff[0].weight, blk["mlp"]["c_fc"]["w"].T)
        ff[0].bias = assign(ff[0].bias, blk["mlp"]["c_fc"]["b"])
        ff[2].weight = assign(ff[2].weight, blk["mlp"]["c_proj"]["w"].T)
        ff[2].bias = assign(ff[2].bias, blk["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, blk["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, blk["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, blk["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, blk["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])  # weight tying


def load_pretrained_gpt2(model_name="gpt2-small (124M)", models_dir="gpt2"):
    """Builds a GPTModel and loads OpenAI's weights. Returns (model, config)."""
    from gpt_download import download_and_load_gpt2

    config = {"vocab_size": 50257, "context_length": 1024, "drop_rate": 0.0,
              "qkv_bias": True, **MODEL_CONFIGS[model_name]}
    model_size = model_name.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir=models_dir)

    model = GPTModel(config)
    load_weights_into_gpt(model, params)
    model.eval()
    return model, config


def load_the_verdict(file_path="the-verdict.txt"):
    if not os.path.exists(file_path):
        url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
               "ch02/01_main-chapter-code/the-verdict.txt")
        urllib.request.urlretrieve(url, file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()
    print("Device:", device)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    # ---- 5.1 Untrained model output ----
    token_ids = generate_text_simple(model, text_to_token_ids("Every effort moves you", tokenizer),
                                     max_new_tokens=10,
                                     context_size=GPT_CONFIG_124M["context_length"])
    print("Untrained output:", token_ids_to_text(token_ids, tokenizer))

    # ---- 5.1 Loss by hand: how probable are the correct next tokens? ----
    inputs = torch.tensor([[16833, 3626, 6100],    # "every effort moves"
                           [40, 1107, 588]])       # "I really like"
    targets = torch.tensor([[3626, 6100, 345],     # " effort moves you"
                            [1107, 588, 11311]])   # " really like chocolate"
    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    print("Predicted:", token_ids_to_text(torch.argmax(probas, dim=-1, keepdim=True)[0].flatten(), tokenizer))
    text_idx = 0
    p1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    text_idx = 1
    p2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    log_probas = torch.log(torch.cat((p1, p2)))
    neg_avg_log_probas = -torch.mean(log_probas)
    print("Manual cross-entropy:", neg_avg_log_probas)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    print("PyTorch cross-entropy:", loss)          # same value, ~10.79
    print("Perplexity:", torch.exp(loss))          # ~48725 = "unsure among ~48k tokens"

    # ---- 5.1 Train/val split on The Verdict ----
    text_data = load_the_verdict()
    print("Characters:", len(text_data), "Tokens:", len(tokenizer.encode(text_data)))

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data, val_data = text_data[:split_idx], text_data[split_idx:]

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(train_data, batch_size=2,
                                        max_length=GPT_CONFIG_124M["context_length"],
                                        stride=GPT_CONFIG_124M["context_length"],
                                        drop_last=True, shuffle=True)
    val_loader = create_dataloader_v1(val_data, batch_size=2,
                                      max_length=GPT_CONFIG_124M["context_length"],
                                      stride=GPT_CONFIG_124M["context_length"],
                                      drop_last=False, shuffle=False)

    model.to(device)
    with torch.no_grad():
        print("Initial train loss:", calc_loss_loader(train_loader, model, device))
        print("Initial val loss:", calc_loss_loader(val_loader, model, device))

    # ---- 5.2 Train (a few minutes on CPU) ----
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs,
        eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer,
    )
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    # Val loss stops improving while train loss keeps falling -> overfitting
    # (expected: the dataset is tiny)

    # ---- 5.3 Temperature scaling demo ----
    vocab = {"closer": 0, "every": 1, "effort": 2, "forward": 3, "inches": 4,
             "moves": 5, "pizza": 6, "toward": 7, "you": 8}
    inverse_vocab = {v: k for k, v in vocab.items()}
    next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
    for T in [1, 0.1, 5]:
        torch.manual_seed(123)
        probs = torch.softmax(next_token_logits / T, dim=0)
        sample = [torch.multinomial(probs, num_samples=1).item() for _ in range(1000)]
        counts = torch.bincount(torch.tensor(sample), minlength=len(vocab))
        print(f"\nTemperature {T}:")
        for i, freq in enumerate(counts):
            print(f"  {freq} x {inverse_vocab[i]}")

    # ---- 5.3 Top-k + temperature generation ----
    torch.manual_seed(123)
    model.eval()
    token_ids = generate(model, text_to_token_ids("Every effort moves you", tokenizer).to(device),
                         max_new_tokens=15, context_size=GPT_CONFIG_124M["context_length"],
                         top_k=25, temperature=1.4)
    print("Top-k sample:", token_ids_to_text(token_ids, tokenizer))

    # ---- 5.4 Save and load (model + optimizer, so training can resume) ----
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, "model_and_optimizer.pth")
    checkpoint = torch.load("model_and_optimizer.pth", map_location=device, weights_only=True)
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # ---- 5.5 Load OpenAI's GPT-2 weights (~500 MB download for 124M) ----
    gpt, cfg = load_pretrained_gpt2("gpt2-small (124M)")
    gpt.to(device)
    torch.manual_seed(123)
    token_ids = generate(gpt, text_to_token_ids("Every effort moves you", tokenizer).to(device),
                         max_new_tokens=25, context_size=cfg["context_length"],
                         top_k=50, temperature=1.5)
    print("GPT-2 output:", token_ids_to_text(token_ids, tokenizer))
