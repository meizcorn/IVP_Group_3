import pandas as pd
import numpy as np
from tqdm import tqdm

from config import TRAIN_CSV, TEST_CSV, TRAIN_IMAGE_DIR, TEST_IMAGE_DIR, IMAGE_SIZE
from preprocessing import load_grayscale_image, preprocess_image
from features import extract_hog_features


##path for training image
def get_train_image_path(image_id, category):
    return TRAIN_IMAGE_DIR / str(category) / f"{image_id}.png"

##path for test image
def get_test_image_path(image_id):
    return TEST_IMAGE_DIR / f"{image_id}.png"


def load_train_dataframe():
    return pd.read_csv(TRAIN_CSV)


def load_test_dataframe():
    return pd.read_csv(TEST_CSV)


##load all training images, preprocess them, extract HOG features
def build_train_features(use_crop=True):
    df = load_train_dataframe()

    X = []    ##feature matrix
    y = []   ##labels
    ids = []    ##image ids

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building train features"):
        image_id = row["Id"]
        category = row["Category"]

        image_path = get_train_image_path(image_id, category)

        image = load_grayscale_image(image_path)
        image = preprocess_image(
            image,
            target_size=IMAGE_SIZE,
            use_crop=use_crop,
            use_normalize=True,
        )

        features = extract_hog_features(image)

        X.append(features)
        y.append(category)
        ids.append(image_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    ids = np.array(ids)

    return X, y, ids



##same but for test, load all test images, preprocess, extract HOG features
def build_test_features(use_crop=True):
    df = load_test_dataframe()

    X_test = []   ##feature matrix
    ids = []    ##ids

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building test features"):
        image_id = row["Id"]
        image_path = get_test_image_path(image_id)

        image = load_grayscale_image(image_path)
        image = preprocess_image(
            image,
            target_size=IMAGE_SIZE,
            use_crop=use_crop,
            use_normalize=True,
        )

        features = extract_hog_features(image)

        X_test.append(features)
        ids.append(image_id)

    X_test = np.array(X_test, dtype=np.float32)
    ids = np.array(ids)

    return X_test, ids