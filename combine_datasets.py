import pandas as pd


def combine_data():
    # Load fake and real datasets (update paths as needed) and add labels
    df_fake = pd.read_csv("data\Fake.csv")
    df_fake["label"] = "0"  # 0 for fake news
    df_real = pd.read_csv("data\True.csv")
    df_real["label"] = "1"  # 1 for real news

    # Combine datasets
    df = pd.concat([df_fake, df_real], ignore_index=True)

    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert to CSV
    df.to_csv("data/combined_dataset.csv", index=False)
    print("Data Combined!\n")


if __name__ == "__main__":
    combine_data()
