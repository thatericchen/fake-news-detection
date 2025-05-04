import kagglehub
import os


# Download dataset and print the path
def download_data():
    path = kagglehub.dataset_download("emineyetm/fake-news-detection-datasets")
    print("Path: ", path)
    print("\nData Downloaded!\n")

    os.makedirs("data", exist_ok=True)

    # Walk the download path and move all .csv files into ./data/
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".csv"):
                src = os.path.join(root, file)
                dst = os.path.join("data", file)

                with open(src, "rb") as f_src:
                    with open(dst, "wb") as f_dst:
                        f_dst.write(f_src.read())

                print(f"Copied: {file} → data/")

    print("\nCSV files copied to ./data\n")


if __name__ == "__main__":
    download_data()
