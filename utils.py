"""Utility functions for project 1."""
import yaml
import os
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress bars
import datetime
import json

from sklearn.metrics import mean_absolute_error, r2_score

IMAGE_SIZE = (300, 300)


def load_config():
    with open("./config.yaml", "r") as file:
        config = yaml.safe_load(file)

    config["data_dir"] = Path(config["data_dir"])

    if config["load_rgb"] is None or config["downsample_factor"] is None:
        raise NotImplementedError("Make sure to set load_rgb and downsample_factor!")

    print(f"[INFO]: Configs are loaded with: \n {config}")
    return config


def load_dataset(config, split="train"):
    labels = pd.read_csv(
        config["data_dir"] / f"{split}_labels.csv", dtype={"ID": str}
    )

    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    )
    feature_dim = feature_dim * 3 if config["load_rgb"] else feature_dim

    images = np.zeros((len(labels), feature_dim))

    # Add progress bar for dataset loading
    print(f"Loading {split} dataset...")
    for idx, (_, row) in enumerate(tqdm(list(labels.iterrows()), desc=f"Loading {split} images")):
        image = Image.open(
            config["data_dir"] / f"{split}_images" / f"{row['ID']}.png"
        )
        if not config["load_rgb"]:
            image = image.convert("L")
        image = image.resize(
            (
                IMAGE_SIZE[0] // config["downsample_factor"],
                IMAGE_SIZE[1] // config["downsample_factor"],
            ),
            resample=Image.BILINEAR,
        )
        image = np.asarray(image).reshape(-1)
        images[idx] = image

    distances = labels["distance"].to_numpy()
    return images, distances, split


def load_test_dataset(config):
    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    )
    feature_dim = feature_dim * 3 if config["load_rgb"] else feature_dim

    images = []
    img_root = os.path.join(config["data_dir"], "test_images")

    # Get list of files before progress bar
    img_files = [f for f in sorted(os.listdir(img_root)) if f.endswith(".png")]
    
    # Add progress bar for test dataset loading
    print("Loading test dataset...")
    for img_file in tqdm(img_files, desc="Loading test images"):
        image = Image.open(os.path.join(img_root, img_file))
        if not config["load_rgb"]:
            image = image.convert("L")
        image = image.resize(
            (
                IMAGE_SIZE[0] // config["downsample_factor"],
                IMAGE_SIZE[1] // config["downsample_factor"],
            ),
            resample=Image.BILINEAR,
        )
        image = np.asarray(image).reshape(-1)
        images.append(image)

    return images


def print_results(gt, pred):
    mae = mean_absolute_error(gt, pred)
    r2 = r2_score(gt, pred)*100
    print(f"MAE: {round(mae, 5)}" + f"|| R2: {round(r2, 3)}")
    return mae, r2


def save_results(pred):
    text = "ID,Distance\n"

    for i, distance in enumerate(pred):
        text += f"{i:03d},{distance}\n"

    with open("prediction.csv", 'w') as f: 
        f.write(text)

