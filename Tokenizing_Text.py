import urllib.request
import os

if not os.path.exists("the-verdict.txt"):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)


with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()
print(raw_text)
print(len(raw_text))

import re
'''
text = "hello jd ajsas A. sjdfnjsd"
result = re.split(r'([,.]|\s)',text)
result =[item for item in result if item.strip()]
print(result)
WE LEARNED THE FUNCTION AND TESTED IT IN RANDOM TEXT
'''
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)
preprocessed =[item for item in preprocessed if item.strip()]
print(len(preprocessed))

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
vocab = {token:integer for integer, token in enumerate(all_words)}


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
'''
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
It had error because in the text we preprocessed, there is no hello. So it gave keyerror 
'''

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
'''Lets see what is happening'''
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
    item.strip() for item in preprocessed if item.strip()
]
        preprocessed = [item if item in self.str_to_int
                    else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
text = "Hello, do you like tea?"
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))
#In this way we can also decode unknown words like <unk>

#Byte-Pair Encoding

from importlib.metadata import version
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.decode(tokenizer.encode("hello world")))
text2= "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace"
print(tokenizer.encode(text2, allowed_special={"<|endoftext|>"}))

#Data Sampling with a sliding window
with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")
#this context size determines how many tokens in an output and help to find the next word

for i in range(1,context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context),"---->",tokenizer.decode([desired]))
#it works well

import torch

#pytorch dataset class
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                 stride=128, shuffle=True, drop_last=True,
                 num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
                            dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers = num_workers)
    return dataloader

dataloader = create_dataloader_v1(raw_text,batch_size=1,max_length = 4,stride = 1,shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
#if we wanna see the secondbatch
second_batch = next(data_iter)
print(second_batch)

dataloader_better = create_dataloader_v1(raw_text,batch_size=8,max_length = 4,stride = 1,shuffle=False)
data_iter = iter(dataloader_better)
inputs, targets = next(data_iter)
print("Inputs:\n",inputs)
print("Targets:\n",targets)

#Creating token embeddings
input_ids = torch.tensor([2,3,5,1])
output_dim = 3
vocab_size = 6
torch.manual_seed(123)
#random seed helped us to get same numbers each time code is running
embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
print(embedding_layer.weight)
#weights will be used in nn training
print(embedding_layer(input_ids))


vocab_size = tokenizer.n_vocab
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
token_embeds = token_embedding_layer(inputs)
print(token_embeds.shape)

context_length = 4
pos_embedding_layer= torch.nn.Embedding(context_length,output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(4))
print(pos_embeddings.shape)
input_embedding = token_embeds+pos_embeddings
print(input_embedding.shape)