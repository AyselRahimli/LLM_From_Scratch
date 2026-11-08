# Chapter 4: Implementing a GPT model from scratch to generate text

import tiktoken
import torch
import torch.nn as nn

from previous_chapters import MultiHeadAttention

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # BPE vocabulary size (tiktoken "gpt2")
    "context_length": 1024,  # max number of input tokens
    "emb_dim": 768,          # embedding dimension
    "n_heads": 12,           # attention heads
    "n_layers": 12,          # transformer blocks
    "drop_rate": 0.1,        # dropout rate
    "qkv_bias": False,       # bias in query/key/value Linear layers
}


# ---------------------------------------------------------------
# 4.2 Layer normalization: mean 0, variance 1 across the embedding dim
# ---------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5                                 # avoids division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))  # trainable
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # trainable

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # biased variance, like GPT-2
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# ---------------------------------------------------------------
# 4.3 GELU activation + feed-forward network
# ---------------------------------------------------------------
class GELU(nn.Module):
    def forward(self, x):
        # tanh approximation used by the original GPT-2
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # expand 768 -> 3072
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # shrink back 3072 -> 768
        )

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------------
# 4.4 Shortcut (residual) connections: fix vanishing gradients
# ---------------------------------------------------------------
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), GELU())
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


# ---------------------------------------------------------------
# 4.5 Transformer block = attention + feed-forward, each with
#     pre-LayerNorm, dropout, and a shortcut connection
# ---------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


# ---------------------------------------------------------------
# 4.6 The full GPT model
# ---------------------------------------------------------------
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds          # (batch, seq_len, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)            # (batch, seq_len, vocab_size)
        return logits


# ---------------------------------------------------------------
# 4.7 Generating text (greedy decoding)
# ---------------------------------------------------------------
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx: (batch, n_tokens) token IDs of the current context
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]   # crop to the supported context size
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]           # only the last time step matters
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    # ---- Batch of two texts ----
    batch = torch.stack([
        torch.tensor(tokenizer.encode("Every effort moves you")),
        torch.tensor(tokenizer.encode("Every day holds a")),
    ], dim=0)
    print(batch)

    # ---- LayerNorm demo ----
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    torch.set_printoptions(sci_mode=False)
    print("Mean:\n", out_ln.mean(dim=-1, keepdim=True))                    # ~0
    print("Variance:\n", out_ln.var(dim=-1, unbiased=False, keepdim=True))  # ~1

    # ---- GELU vs ReLU plot ----
    try:
        import matplotlib.pyplot as plt
        gelu, relu = GELU(), nn.ReLU()
        x = torch.linspace(-3, 3, 100)
        plt.figure(figsize=(8, 3))
        for i, (y, label) in enumerate(zip([gelu(x), relu(x)], ["GELU", "ReLU"]), 1):
            plt.subplot(1, 2, i)
            plt.plot(x, y)
            plt.title(f"{label} activation function")
            plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

    # ---- FeedForward keeps the shape ----
    ffn = FeedForward(GPT_CONFIG_124M)
    print(ffn(torch.rand(2, 3, 768)).shape)  # torch.Size([2, 3, 768])

    # ---- Shortcut connections: compare gradients ----
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])
    torch.manual_seed(123)
    print("Without shortcut:")
    print_gradients(ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False), sample_input)
    torch.manual_seed(123)
    print("With shortcut:")
    print_gradients(ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True), sample_input)

    # ---- Transformer block keeps the shape ----
    torch.manual_seed(123)
    block = TransformerBlock(GPT_CONFIG_124M)
    x = torch.rand(2, 4, 768)
    print("Block input:", x.shape, "output:", block(x).shape)

    # ---- Full GPT model ----
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    out = model(batch)
    print("Output shape:", out.shape)  # torch.Size([2, 4, 50257])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")  # 163,009,536
    # GPT-2 reuses the token embedding as the output layer ("weight tying"):
    params_tied = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Trainable params with weight tying: {params_tied:,}")  # 124,412,160
    total_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    print(f"Model size: {total_size_mb:.2f} MB")      # ~621.83 MB

    # Exercise 4.1: params in feed-forward vs attention modules
    ff_params = sum(p.numel() for p in block.ff.parameters())
    att_params = sum(p.numel() for p in block.att.parameters())
    print(f"FeedForward params per block: {ff_params:,}")   # 4,722,432
    print(f"Attention params per block: {att_params:,}")    # 2,360,064

    # Exercise 4.2: bigger GPT-2 configs
    for name, (emb, layers, heads) in {"medium": (1024, 24, 16),
                                       "large": (1280, 36, 20),
                                       "xl": (1600, 48, 25)}.items():
        cfg = {**GPT_CONFIG_124M, "emb_dim": emb, "n_layers": layers, "n_heads": heads}
        n = sum(p.numel() for p in GPTModel(cfg).parameters())
        print(f"gpt2-{name}: {n:,} params")

    # ---- Generate text (untrained -> gibberish) ----
    start_context = "Hello, I am"
    encoded_tensor = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0)
    model.eval()  # turn off dropout
    out = generate_text_simple(model, encoded_tensor, max_new_tokens=6,
                               context_size=GPT_CONFIG_124M["context_length"])
    print("Output:", out)
    print(tokenizer.decode(out.squeeze(0).tolist()))
