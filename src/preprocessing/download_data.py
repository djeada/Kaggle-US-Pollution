# use path to get home directory
from pathlib import Path

DATASET_URL = "sogun3/uspollution"


def assert_prerequisites() -> None:
    """
    Asserts that all prerequisites are met in order to use kaggle API.
    """
    home = Path.home()
    # check if ~/.kaggle exists
    if not (home / ".kaggle").exists():
        print("Creating ~/.kaggle directory")
        (home / ".kaggle").mkdir()

    if not (home / ".kaggle/kaggle.json").exists():
        print(
            "You need to authenticate with kaggle first in order to download the data"
        )
        print("Please follow the instructions here: ")
        # input your username and key
        username = input("Enter your Kaggle username: ")
        key = input("Enter your Kaggle key: ")
        (home / ".kaggle/kaggle.json").write_text(
            f'{{"username":"{username}","key":"{key}"}}'
        )


def download_dataset(dataset_path: Path) -> None:
    """
    Downloads the dataset from kaggle.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()

    try:
        api.authenticate()
    except:
        print("Authentication failed")
        return

    print("Downloading data")
    api.dataset_download_files(DATASET_URL, path=dataset_path.parent, unzip=True)
    print("Done")


def assert_dataset_exists(dataset_path: Path) -> None:
    """
    Asserts that the dataset exists.
    If not, it downloads the dataset.
    """
    if dataset_path.exists():
        return

    print("Dataset not found")
    assert_prerequisites()
    download_dataset(dataset_path)
