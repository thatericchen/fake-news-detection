import combine_datasets
import download_dataset
import preprocessing
import subprocess
import os
import argparse
from run_model import run_model
from visualization import visualize

parser = argparse.ArgumentParser()
parser.add_argument("--force-download", action="store_true")
parser.add_argument("--force-combine", action="store_true")
parser.add_argument("--force-preprocess", action="store_true")
args = parser.parse_args()
# Usage: python data_pipeline.py --force-download
models = ["logisticregression", "naivebayes", "randomforest"]
model_file_name = {
    "logisticregression": "logistic_regression_model",
    "naivebayes": "naive_bayes_model",
    "randomforest": "random_forest_model",
}
model_type = " "


def pipeline():
    print("Pipeline Active!")


if __name__ == "__main__":
    pipeline()
    # Changes kagglehub download path to /data folder
    result = subprocess.run(
        ["./env_script.sh"], capture_output=True, text=True, shell=True
    )

    # Download if not already downloaded
    if args.force_download or not os.path.exists("data/"):
        if args.force_download:
            print("Redownloading dataset...")
        download_dataset.download_data()

    # Combine if not already combined
    if args.force_combine or not os.path.exists("data/combined_dataset.csv"):
        if args.force_combine:
            print("Recombining data...")
        combine_datasets.combine_data()

    # Preprocess if not already preprocessed
    if args.force_preprocess or not os.path.exists("data/processed"):
        if args.force_preprocess:
            print("Reprocessing data...")
        os.makedirs("data/processed", exist_ok=True)
        preprocessing.preprocess_data()

    # Ask for which model to use
    while model_type not in models:
        model_type = (
            input(
                "Enter which model you want to use: ( logisticregression / naivebayes / randomforest )\n> "
            )
            .strip()
            .lower()
        )
        model_type = model_type.replace(" ", "").lower()

    if not os.path.exists(f"data/{model_file_name[model_type]}.pkl"):
        run_model(model_type)
        print(f"Finished running model: {model_type}...")

    visualize(model_type)
    print(f"Visualizing for model: {model_type}...")
