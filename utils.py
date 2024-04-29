"""Utility functions for project 1."""
import yaml
import os
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image

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
        config["data_dir"] / f"{split}_labels.csv", dtype={"image_name": str}
    )

    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    )
    feature_dim = feature_dim * 3 if config["load_rgb"] else feature_dim

    images = np.zeros((len(labels), feature_dim))

    idx = 0
    for _, row in labels.iterrows():
        image = Image.open(
            config["data_dir"] / f"{split}_images" / f"{row['image_name']}.png"
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
        idx += 1

    distances = labels["distance"].to_numpy()
    return images, distances


def load_private_test_dataset(config):
    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    )
    feature_dim = feature_dim * 3 if config["load_rgb"] else feature_dim

    images = []
    img_root = os.path.join(config["data_dir"], "private_test_images")

    for img_file in sorted(os.listdir(img_root)):
        if img_file.endswith(".png"):
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
    print(f"MAE: {round(mean_absolute_error(gt, pred)*100, 3)}")
    print(f"R2: {round(r2_score(gt, pred)*100, 3)}")

def save_results(pred):
    text = "image_name,distance\n"

    for i, distance in enumerate(pred):
        text += f"{i:06d},{distance}\n"

    with open("prediction.csv", 'w') as f: 
        f.write(text)
