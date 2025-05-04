from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    PrecisionRecallDisplay,
    average_precision_score
)
import matplotlib.pyplot as plt
import pandas as pd
import joblib

fig, axis = plt.subplots(1, 4, figsize=(24, 12))

def plot_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = (
        pd.DataFrame(report)
        .transpose()
        .loc[["0", "1"], ["precision", "recall", "f1-score"]]
    )
    df.index = ["Real", "Fake"]

    ax = df.plot(
        kind="bar", ax=axis[3], figsize=(16, 6), ylim=(0, 1), rot=0, legend=True
    )
    ax.set_title("Precision, Recall, and F1 Score by Class", fontsize=14)
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()


def visualize(model_type="logisticregression"):
    X_train = joblib.load("data/processed/X_train.pkl")
    X_test = joblib.load("data/processed/X_test.pkl")
    X_val = joblib.load("data/processed/X_val.pkl")
    y_train = joblib.load("data/processed/y_train.pkl")
    y_test = joblib.load("data/processed/y_test.pkl")
    y_val = joblib.load("data/processed/y_val.pkl")

    if model_type == "logisticregression":
        model = joblib.load("data/logistic_regression_model.pkl")
        X_test_aug = model.bias_augment(X_test.toarray())
        y_scores = model.predict_probs(X_test_aug, model.weights)
        y_pred = model.predict_labels(y_scores, model.threshold)
    elif model_type == "naivebayes":
        model = joblib.load("data/naive_bayes_model.pkl")
        y_scores = model.predict_probs(X_test)[:, 1]
        y_pred = model.predict(X_test)
    elif model_type == "randomforest":
        model = joblib.load("data/random_forest_model.pkl")
        y_scores = model.predict_probs(X_test)[:, 1]
        y_pred = model.predict(X_test)

    # Plot 1: Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axis[0], display_labels=["Real", "Fake"], colorbar=False)
    axis[0].set_title("Confusion Matrix")

    # Plot 2: ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_scores, ax=axis[1])
    axis[1].set_title("ROC Curve")

    # PLot 3: PR Curve
    pr_auc = average_precision_score(y_test, y_scores)
    PrecisionRecallDisplay.from_predictions(y_test, y_scores, ax=axis[2])
    axis[2].set_title(f"PR Curve (AUC = {pr_auc:.2f})")

    # Plot 4: Raw Metrics
    plot_classification_report(y_test, y_pred)
    axis[3].set_title("Metrics")

    # Plot grouped bars
    plt.tight_layout()
    plt.show()
