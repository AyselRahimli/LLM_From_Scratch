# Chapter 7: Fine-tuning to follow instructions

import json
import os
import time
import urllib.request
from functools import partial

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ch05_pretraining import (calc_loss_loader, generate, load_pretrained_gpt2, plot_losses,
                              text_to_token_ids, token_ids_to_text, train_model_simple)
from previous_chapters import get_device


# ---------------------------------------------------------------
# 7.2 Download the 1,100 instruction-response examples
# ---------------------------------------------------------------
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# Alpaca-style prompt template
def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


# ---------------------------------------------------------------
# 7.3 Dataset: pre-tokenize full prompt + response
# ---------------------------------------------------------------
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            full_text = format_input(entry) + f"\n\n### Response:\n{entry['output']}"
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------
# 7.3 Custom collate function:
#   - pad each batch only to ITS longest example (not the whole dataset)
#   - targets = inputs shifted by 1
#   - all padding tokens in targets except the first become -100,
#     which cross_entropy ignores (so padding doesn't affect the loss)
# ---------------------------------------------------------------
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                      allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]  # one <|endoftext|> so the model learns to stop
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index  # keep the first pad as the "end" signal

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)


# ---------------------------------------------------------------
# 7.8 Evaluate responses with a bigger local model via Ollama
#     Install Ollama (https://ollama.com), then run:  ollama pull llama3
# ---------------------------------------------------------------
def check_if_running(process_name="ollama"):
    import psutil
    return any(process_name in (proc.info["name"] or "") for proc in psutil.process_iter(["name"]))


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": 123, "temperature": 0, "num_ctx": 2048},  # deterministic
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:  # Ollama streams one JSON object per line
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_data += json.loads(line)["message"]["content"]
    return response_data


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
    return scores


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()
    print("Device:", device)

    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
           "ch07/01_main-chapter-code/instruction-data.json")
    data = download_and_load_file("instruction-data.json", url)
    print("Number of entries:", len(data))

    model_input = format_input(data[50])
    print(model_input + f"\n\n### Response:\n{data[50]['output']}")

    # ---- Split 85% / 5% / 10% ----
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    print("Train:", len(train_data), "Val:", len(val_data), "Test:", len(test_data))

    # ---- Collate demo ----
    inputs, targets = custom_collate_fn([[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]])
    print(inputs)
    print(targets)  # note the -100 entries

    # ---- DataLoaders ----
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
    num_workers, batch_size = 0, 8
    torch.manual_seed(123)
    train_loader = DataLoader(InstructionDataset(train_data, tokenizer), batch_size=batch_size,
                              collate_fn=customized_collate_fn, shuffle=True, drop_last=True,
                              num_workers=num_workers)
    val_loader = DataLoader(InstructionDataset(val_data, tokenizer), batch_size=batch_size,
                            collate_fn=customized_collate_fn, shuffle=False, drop_last=False,
                            num_workers=num_workers)

    # ---- Load GPT-2 medium (355M). 124M is too small for good results here.
    #      If your machine struggles, switch to "gpt2-small (124M)". ----
    model, cfg = load_pretrained_gpt2("gpt2-medium (355M)")
    model.to(device)

    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    token_ids = generate(model, text_to_token_ids(input_text, tokenizer).to(device),
                         max_new_tokens=35, context_size=cfg["context_length"], eos_id=50256)
    print("Before fine-tuning:", token_ids_to_text(token_ids, tokenizer)[len(input_text):].strip())

    with torch.no_grad():
        print("Initial train loss:", calc_loss_loader(train_loader, model, device, num_batches=5))
        print("Initial val loss:", calc_loss_loader(val_loader, model, device, num_batches=5))

    # ---- 7.6 Fine-tune (~15 min on M3 CPU for 355M, ~1 min on a GPU) ----
    start_time = time.time()
    torch.manual_seed(123)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    num_epochs = 2
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs,
        eval_freq=5, eval_iter=5, start_context=format_input(val_data[0]), tokenizer=tokenizer,
    )
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, filename="instruct-loss-plot.pdf")

    # ---- 7.7 Generate responses for the test set ----
    model.eval()
    torch.manual_seed(123)
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
        token_ids = generate(model, text_to_token_ids(input_text, tokenizer).to(device),
                             max_new_tokens=256, context_size=cfg["context_length"], eos_id=50256)
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
        test_data[i]["model_response"] = response_text
        if i < 3:
            print(input_text)
            print(f"\nCorrect response:\n>> {entry['output']}")
            print(f"\nModel response:\n>> {response_text}")
            print("-------------------------------------")

    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)

    torch.save(model.state_dict(), "gpt2-medium355M-sft.pth")

    # ---- 7.8 Score the responses with Llama 3 (needs Ollama running) ----
    if check_if_running("ollama"):
        print(query_model("What do Llamas eat?"))
        scores = generate_model_scores(test_data, "model_response")
        print(f"Number of scores: {len(scores)} of {len(test_data)}")
        print(f"Average score: {sum(scores) / len(scores):.2f}")  # ~50 in the book
    else:
        print("Ollama not running. Start it with `ollama serve` and run `ollama pull llama3`.")
