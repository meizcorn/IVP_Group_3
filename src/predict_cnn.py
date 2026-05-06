import pandas as pd
import torch
import numpy as np

from CNN_model import DigitCNN
from data_loader import load_test_dataframe, get_test_image_path
from preprocessing import load_grayscale_image, preprocess_image
from config import IMAGE_SIZE, OUTPUT_DIR


def main():
    model = DigitCNN()
    model.load_state_dict(torch.load("models/cnn_model.pth", map_location="cpu"))
    model.eval()

    df = load_test_dataframe()

    X = []
    test_ids = []

    for _, row in df.iterrows():
        image_id = row["Id"]
        image_path = get_test_image_path(image_id)

        image = load_grayscale_image(image_path)
        image = preprocess_image(
            image,
            target_size=IMAGE_SIZE,
            use_crop=True,
            use_normalize=True
        )

        X.append(image)
        test_ids.append(image_id)

    X = np.array(X, dtype=np.float32)
    X = torch.tensor(X).unsqueeze(1)

    with torch.no_grad():
        outputs = model(X)
        _, predictions = torch.max(outputs, 1)

    submission = pd.DataFrame({
        "Id": test_ids,
        "Category": predictions.numpy().astype(int)
    })

    submission_path = OUTPUT_DIR / "submission_cnn.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Saved CNN submission to: {submission_path}")
    print(submission.head())


if __name__ == "__main__":
    main()