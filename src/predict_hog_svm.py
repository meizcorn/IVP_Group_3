import pandas as pd
import joblib

from config import HOG_SVM_MODEL_PATH, OUTPUT_DIR
from data_loader import build_test_features


def main():
    print(f"Loading model from: {HOG_SVM_MODEL_PATH}")
    model = joblib.load(HOG_SVM_MODEL_PATH)

    print("\nLoading test data and extracting HOG features...")
    X_test, test_ids = build_test_features(use_crop=True)

    print(f"\nTest feature matrix shape: {X_test.shape}")

    print("\nPredicting test labels...")
    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        "Id": test_ids,
        "Category": predictions.astype(int),
    })

    submission_path = OUTPUT_DIR / "submission_hog_svm.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\nSaved submission to: {submission_path}")
    print(submission.head())


if __name__ == "__main__":
    main()