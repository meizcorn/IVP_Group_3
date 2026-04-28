from pathlib import Path

#root directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "iivp-2026-challenge"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

TRAIN_IMAGE_DIR = DATA_DIR / "train" / "train"
TEST_IMAGE_DIR = DATA_DIR / "test" / "test"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# Model paths
HOG_SVM_MODEL_PATH = MODEL_DIR / "hog_svm_model.joblib"



# Image settings
IMAGE_SIZE = (32, 32)

# HOG settings
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"


# Training settings
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
