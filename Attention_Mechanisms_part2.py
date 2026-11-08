# Chapter 3 (continued): Self-attention -> Causal attention -> Multi-head attention
# Continues from Attention_Mechanisms.py (Raschka, "Build a Large Language Model (From Scratch)")

import torch
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)
d_in = inputs.shape[1]  # 3
d_out = 2

# ---------------------------------------------------------------
# 1) SelfAttention_v1 (FIXED version of yours)
#    forward() must use `x` and `self.W_*`, not the global variables.
# ---------------------------------------------------------------
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys    = x @ self.W_key
        queries = x @ self.W_query
        values  = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

# ---------------------------------------------------------------
# 2) SelfAttention_v2: use nn.Linear instead of raw Parameters
#    nn.Linear has better weight initialization -> more stable training.
#    (nn.Linear stores weight as (d_out, d_in), i.e. transposed)
# ---------------------------------------------------------------
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# Exercise 3.1: copy v2's weights into v1 -> both give the same output
sa_v1.W_query.data = sa_v2.W_query.weight.T
sa_v1.W_key.data   = sa_v2.W_key.weight.T
sa_v1.W_value.data = sa_v2.W_value.weight.T
print("v1 == v2:", torch.allclose(sa_v1(inputs), sa_v2(inputs)))
print("-------------------------------------")

# ---------------------------------------------------------------
# 3) Causal (masked) attention
#    A token may only attend to itself and PREVIOUS tokens,
#    otherwise the model could "cheat" by looking at future words.
# ---------------------------------------------------------------
queries = sa_v2.W_query(inputs)
keys    = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

context_length = attn_scores.shape[0]

# Method A: zero out the upper triangle, then re-normalize rows
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
masked_simple = attn_weights * mask_simple
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# Method B (more efficient): put -inf above the diagonal BEFORE softmax.
# exp(-inf) = 0, so softmax gives those positions weight 0 automatically.
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)  # same result as Method A

# ---------------------------------------------------------------
# 4) Dropout on attention weights (only active in training mode)
#    Remaining values are scaled by 1/(1-p) to keep the overall sum similar.
# ---------------------------------------------------------------
torch.manual_seed(123)
dropout = nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))       # ~half are 0, the rest are 2.0
print(dropout(attn_weights))
print("-------------------------------------")

# ---------------------------------------------------------------
# 5) Compact CausalAttention class (supports batches)
# ---------------------------------------------------------------
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)  # torch.Size([2, 6, 3]) -> (batch, num_tokens, d_in)

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # register_buffer: moves to GPU with the model, but is NOT trained
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # transpose(1, 2) instead of .T because of the batch dimension
        attn_scores = queries @ keys.transpose(1, 2)
        # [:num_tokens] handles inputs shorter than context_length
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)  # torch.Size([2, 6, 2])
print("-------------------------------------")

# ---------------------------------------------------------------
# 6) Multi-head attention, version 1: stack several CausalAttention heads
#    Each head learns different relationships; outputs are concatenated.
# ---------------------------------------------------------------
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)  # torch.Size([2, 6, 4]) = num_heads * d_out

# Exercise 3.2: to get a 2-dim output with 2 heads, use d_out=1 per head
mha_ex = MultiHeadAttentionWrapper(d_in, 1, context_length, 0.0, num_heads=2)
print("Exercise 3.2 shape:", mha_ex(batch).shape)  # torch.Size([2, 6, 2])
print("-------------------------------------")

# ---------------------------------------------------------------
# 7) Efficient MultiHeadAttention with weight splits
#    One big matrix multiply, then split into heads via view/transpose.
#    This is the version used in the GPT model (Chapter 4).
# ---------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # combines head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys    = self.W_key(x)     # (b, num_tokens, d_out)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # Split last dim: d_out -> (num_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys    = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values  = values.transpose(1, 2)

        # Attention scores for every head at once: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Merge heads back: (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

# Small demo of batched matmul that the class above relies on
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],
                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])  # (1, 2, 3, 4)
print(a @ a.transpose(2, 3))  # per-head scores, shape (1, 2, 3, 3)

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)  # torch.Size([2, 6, 2])

# Exercise 3.3: GPT-2 small sized attention module
mha_gpt2 = MultiHeadAttention(d_in=768, d_out=768, context_length=1024,
                              dropout=0.0, num_heads=12)
num_params = sum(p.numel() for p in mha_gpt2.parameters())
print(f"GPT-2 small attention params: {num_params:,}")  # 2,360,064
