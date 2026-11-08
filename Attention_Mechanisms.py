#Simple self-attention mechanisms without trainable weights

import torch
inputs = torch.tensor(
    [
    [0.43,0.15,0.89],
    [0.55,0.23,0.58],
    [0.87,0.57,0.64],
    [0.22, 0.58,0.33],
    [0.77,0.25,0.1],
    [0.05,0.8,0.55]]
)

input_query =inputs[0]

query = inputs[1]
attn_scores = torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
    attn_scores[i] = torch.dot(x_i,input_query)
print(attn_scores)

attn_weights_tmp = attn_scores/(attn_scores.sum())
print(attn_weights_tmp)
print(attn_weights_tmp.sum()) #it shows that is is normalized

def softmax_naive(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)

print(softmax_naive((attn_scores)))
#or it can be computed like this
attn_weights = torch.softmax(attn_scores,dim=0)

context_vec =torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec +=attn_weights[i]*inputs[i]
print(context_vec)
print("------------------------------------------------")
#Computing attention weights for all input tokens
query = inputs[1]
attn_scores_2 = torch.empty(6,6)

for i,x_i in enumerate(inputs):
    for j,x_j in enumerate(inputs):
        attn_scores_2[i,j] = torch.dot(x_i,x_j)
print(attn_scores_2)

attn_scores_2 = inputs @ inputs.T

#step by step computing attention
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in,d_out))
W_key = torch.nn.Parameter(torch.rand(d_in,d_out))
W_value = torch.nn.Parameter(torch.rand(d_in,d_out))

query_2 = x_2 @ W_query
print(query_2)
keys = inputs @ W_key
values = inputs @ W_value
print(keys)

keys_2 = keys[1]
attn_scores_22  = torch.dot(query_2,keys_2)
attn_scores_22  = query_2 @ keys.T
print(attn_scores_22)

d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2/d_k**0.5,dim = -1)
print(attn_weights_2)
print(f"SUM: {attn_weights_2.sum()}")

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
print("-------------------------------------")

#Compact Self attention class
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self,d_in,d_out):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in,d_out))
        self.W_key = torch.nn.Parameter(torch.rand(d_in,d_out))
        self.W_value = torch.nn.Parameter(torch.rand(d_in,d_out))
    def forward(self,x):
        keys = inputs@W_key
        queries = inputs @W_query
        values = inputs @W_value

        attn_scores = queries @keys.T
        attn_weights = torch.softmax(attn_scores/d_k**0.5,dim=-1)
        context_vec = attn_weights @ values
        return context_vec

sa_v1 = SelfAttention_v1(d_in,d_out)
print(sa_v1(inputs))

