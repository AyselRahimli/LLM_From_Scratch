# Chapter 6: Fine-tuning for classification (SMS spam detection)

import os
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from ch05_pretraining import load_pretrained_gpt2
from previous_chapters import get_device


# ---------------------------------------------------------------
# 6.2 Download and prepare the dataset
# ---------------------------------------------------------------
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = "sms_spam_collection.zip"
EXTRACTED_PATH = "sms_spam_collection"
DATA_FILE_PATH = Path(EXTRACTED_PATH) / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(url=URL, zip_path=ZIP_PATH,
                                 extracted_path=EXTRACTED_PATH, data_file_path=DATA_FILE_PATH):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download.")
        return
    with urllib.request.urlopen(url) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    os.rename(Path(extracted_path) / "SMSSpamCollection", data_file_path)
    print(f"Downloaded and saved as {data_file_path}")


def create_balanced_dataset(df):
    # There are far more "ham" than "spam" -> undersample ham to match spam count
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]])


def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # shuffle
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:validation_end], df[validation_end:]


def prepare_csv_files():
    download_and_unzip_spam_data()
    df = pd.read_csv(DATA_FILE_PATH, sep="\t", header=None, names=["Label", "Text"])
    print(df["Label"].value_counts())
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)  # 70/10/20
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)


# ---------------------------------------------------------------
# 6.3 Dataset + DataLoaders (pad every message to the same length)
# ---------------------------------------------------------------
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [t[:self.max_length] for t in self.encoded_texts]

        # pad with <|endoftext|> (ID 50256)
        self.encoded_texts = [t + [pad_token_id] * (self.max_length - len(t))
                              for t in self.encoded_texts]

    def __getitem__(self, index):
        return (torch.tensor(self.encoded_texts[index], dtype=torch.long),
                torch.tensor(self.data.iloc[index]["Label"], dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max(len(t) for t in self.encoded_texts)


def create_spam_loaders(tokenizer, batch_size=8, num_workers=0):
    if not Path("train.csv").exists():
        prepare_csv_files()
    train_dataset = SpamDataset("train.csv", tokenizer, max_length=None)
    val_dataset = SpamDataset("validation.csv", tokenizer, max_length=train_dataset.max_length)
    test_dataset = SpamDataset("test.csv", tokenizer, max_length=train_dataset.max_length)

    torch.manual_seed(123)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                            drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                             drop_last=False)
    return train_loader, val_loader, test_loader, train_dataset.max_length


# ---------------------------------------------------------------
# 6.5 Turn GPT-2 into a classifier
# ---------------------------------------------------------------
def build_spam_classifier(device, num_classes=2):
    model, cfg = load_pretrained_gpt2("gpt2-small (124M)")

    # Freeze everything...
    for param in model.parameters():
        param.requires_grad = False

    # ...replace the 50257-way output head with a 2-way head (new layer = trainable)
    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(cfg["emb_dim"], num_classes)

    # ...and unfreeze the last transformer block + final LayerNorm (better accuracy)
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    return model.to(device), cfg


# ---------------------------------------------------------------
# 6.6 Loss and accuracy: only the LAST token's output is used,
#     because thanks to causal attention it has "seen" the whole message
# ---------------------------------------------------------------
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)
        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


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


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# ---------------------------------------------------------------
# 6.7 Fine-tuning loop (tracks examples seen instead of tokens)
# ---------------------------------------------------------------
def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader,
                                                      device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy * 100:.2f}% | "
              f"Validation accuracy: {val_accuracy * 100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


# ---------------------------------------------------------------
# 6.8 Use the classifier on new text
# ---------------------------------------------------------------
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    if max_length is None:
        max_length = supported_context_length
    max_length = min(max_length, supported_context_length)
    input_ids = input_ids[:max_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()

    train_loader, val_loader, test_loader, max_length = create_spam_loaders(tokenizer)
    print(f"{len(train_loader)} training batches, {len(val_loader)} validation batches, "
          f"{len(test_loader)} test batches. Max length: {max_length}")

    model, cfg = build_spam_classifier(device)

    print("Output shape for 4 tokens:", model(torch.tensor([tokenizer.encode("Do you have time")]).to(device)).shape)  # (1, 4, 2)

    torch.manual_seed(123)
    print(f"Training accuracy (before): {calc_accuracy_loader(train_loader, model, device, 10) * 100:.2f}%")
    print(f"Validation accuracy (before): {calc_accuracy_loader(val_loader, model, device, 10) * 100:.2f}%")

    # ---- Fine-tune (~6 min on a laptop CPU, seconds on GPU) ----
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

    print(f"Training accuracy: {calc_accuracy_loader(train_loader, model, device) * 100:.2f}%")
    print(f"Validation accuracy: {calc_accuracy_loader(val_loader, model, device) * 100:.2f}%")
    print(f"Test accuracy: {calc_accuracy_loader(test_loader, model, device) * 100:.2f}%")  # ~95%

    # ---- Try it ----
    text_1 = ("You are a winner you have been specially selected to receive $1000 cash "
              "or a $2000 award.")
    text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
    print(text_1, "->", classify_review(text_1, model, tokenizer, device, max_length=max_length))
    print(text_2, "->", classify_review(text_2, model, tokenizer, device, max_length=max_length))

    # ---- Save / load ----
    torch.save(model.state_dict(), "review_classifier.pth")
    model.load_state_dict(torch.load("review_classifier.pth", map_location=device, weights_only=True))
