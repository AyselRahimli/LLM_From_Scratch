# Downloads OpenAI's GPT-2 weights (TensorFlow checkpoint) and converts them
# into a nested dict of NumPy arrays. Used in Chapters 5, 6, 7 and Appendix E.
# Requires: tensorflow, numpy, tqdm (all in your requirements file)

import json
import os
import urllib.request

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe",
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        download_file(f"{base_url}/{model_size}/{filename}", os.path.join(model_dir, filename))

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    with open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8") as f:
        settings = json.load(f)
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params


def download_file(url, destination):
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    tmp_path = destination + ".part"  # so an interrupted download isn't mistaken for a finished one
    with urllib.request.urlopen(url) as response:
        file_size = int(response.headers.get("Content-Length", 0))
        with tqdm(total=file_size, unit="iB", unit_scale=True,
                  desc=os.path.basename(url)) as bar, open(tmp_path, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp_path, destination)


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        parts = name.split("/")[1:]  # drop the leading "model/"

        target_dict = params
        if parts[0].startswith("h"):  # e.g. "h0", "h11" = transformer block number
            target_dict = params["blocks"][int(parts[0][1:])]

        for key in parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})
        target_dict[parts[-1]] = variable_array

    return params
