import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    HOG_SVM_MODEL_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    VALIDATION_SIZE,
)
from data_loader import build_train_features
from evaluate import print_classification_results, plot_confusion_matrix



##train the simple HOG + SVM baseline
##use StandardScaler as it is important for SVM because HOG features can have different value distributions
def train_baseline_svm(X_train, y_train):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=False,
                random_state=RANDOM_STATE,
            )),
        ]
    )

    model.fit(X_train, y_train)

    return model

#### IDK TO BE LOOKED AT could potentially be better???
##train SVM with gridsearch? -> maybe improve accuracy? can also take longer tho

# def train_tuned_svm(X_train, y_train):
#     pipeline = Pipeline(
#         steps=[
#             ("scaler", StandardScaler()),
#             ("svm", SVC(
#                 kernel="rbf",
#                 probability=False,
#                 random_state=RANDOM_STATE,
#             )),
#         ]
#     )
#
#     param_grid = {
#         "svm__C": [1, 5, 10, 20],
#         "svm__gamma": ["scale", 0.01, 0.001],
#     }
#
#     grid_search = GridSearchCV(
#         estimator=pipeline,
#         param_grid=param_grid,
#         scoring="accuracy",
#         cv=3,
#         verbose=2,
#         n_jobs=-1,
#     )
#
#     grid_search.fit(X_train, y_train)
#     print("\nBest parameters:")
#     print(grid_search.best_params_)
#     print("\nBest cross-validation accuracy:")
#     print(grid_search.best_score_)
#
#     return grid_search.best_estimator_


def main():
    print("Loading training data and extracting HOG features...")
    X, y, ids = build_train_features(use_crop=True)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nTraining SVM model...")

    model = train_baseline_svm(X_train, y_train)


    print("\nEvaluating on validation set...")
    y_pred = model.predict(X_valid)

    print_classification_results(y_valid, y_pred)

    confusion_matrix_path = OUTPUT_DIR / "confusion_matrix_hog_svm.png"
    plot_confusion_matrix(y_valid, y_pred, save_path=confusion_matrix_path)

    print("\nTraining final model on all training data...")
    final_model = train_baseline_svm(X, y)

    print(f"\nSaving final model to: {HOG_SVM_MODEL_PATH}")
    joblib.dump(final_model, HOG_SVM_MODEL_PATH)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()