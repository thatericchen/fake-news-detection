import joblib
import numpy as np
from FakeNewsClassifier import FakeNewsClassifier
from RandomForestModel import RandomForestFakeNewsClassifier
from naivebayesmodel import MultinomialNBFakeNewsClassifier


def load_data():
    print("Retrieving data......")
    try:
        X_train = joblib.load("data/processed/X_train.pkl")
        y_train = joblib.load("data/processed/y_train.pkl")
        X_val = joblib.load("data/processed/X_val.pkl")
        y_val = joblib.load("data/processed/y_val.pkl")
        X_test = joblib.load("data/processed/X_test.pkl")
        y_test = joblib.load("data/processed/y_test.pkl")

        if hasattr(y_train, "values"):
            print("Converting pandas Series to numpy arrays...")
            y_train = y_train.values
            y_val = y_val.values if hasattr(y_val, "values") else y_val
            y_test = y_test.values if hasattr(y_test, "values") else y_test

        if hasattr(X_train, "toarray"):
            print("Converting sparse matrices to dense arrays...")
            X_train = X_train.toarray()
            X_val = X_val.toarray() if hasattr(X_val, "toarray") else X_val
            X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

        if len(y_train.shape) == 1:
            print("Reshaping y_train...")
            y_train = y_train.reshape(-1, 1)
        if len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)

        print("Data loaded successfully!")

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test, True

    except FileNotFoundError as e:
        print(f"Could not find data. Please check file paths. Error: {e}")
    except Exception as e:
        print(f"An error has occured!: {e}")

    return None, None, None, None, None, None, False


def run_model(model_type):
    if model_type == "logisticregression":
        X_train, y_train, X_val, y_val, X_test, y_test, successful = load_data()
        if not successful:
            print("Data loading unsuccessful. Exiting.")
            return
        print("Running model:")
        model = FakeNewsClassifier(learning_rate=0.05, epochs=800, threshold=0.5)
        learned_weights = model.fit(X_train, y_train, X_val, y_val)
        test_loss, test_acc = model.evaluate(X_test, y_test, learned_weights)
        joblib.dump(model, "data/logistic_regression_model.pkl")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc: {test_acc:.4f}")
        return
    elif model_type == "naivebayes":
        X_train, y_train, X_val, y_val, X_test, y_test, successful = load_data()
        if not successful:
            print("Data loading unsuccessful. Exiting.")
            return

        print("Running Naive Bayes model:")
        model = MultinomialNBFakeNewsClassifier(alpha=1.0)
        model.fit(X_train, y_train)

        # Training evaluation
        model.evaluate(X_train, y_train, "Training")

        # Validation evaluation
        model.evaluate(X_val, y_val, "Validation")

        # Test evaluation
        model.evaluate(X_test, y_test, "Test")

        joblib.dump(model, "data/naive_bayes_model.pkl")
        return
    elif model_type == "randomforest":
        X_train, y_train, X_val, y_val, X_test, y_test, successful = load_data()
        if not successful:
            print("Data loading unsuccessful. Exiting.")
            return
        y_train_flat = y_train.flatten()

        print("Running model:")
        model = RandomForestFakeNewsClassifier(
            num_trees=100, max_depth=50, max_features=0.25
        )
        model.fit(X_train, y_train)

        model.confusion_matrix(X_test, y_test)

        # validation predictions
        print(f"Calculating validation accuracy")
        val_preds = model.predict(X_val)
        val_acc = np.mean(val_preds == y_val.flatten())
        print(f"Validation Acc: {val_acc:.4f}")

        # test predictions
        print("Calculating test accuracy")
        test_preds = model.predict(X_test)
        test_acc = np.mean(test_preds == y_test.flatten())
        print(f"Test Acc: {test_acc:.4f}")

        joblib.dump(model, "data/random_forest_model.pkl")
        return
