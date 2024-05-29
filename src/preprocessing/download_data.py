from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import logging

logger = logging.getLogger(__name__)


def assert_prerequisites() -> None:
    """
    Asserts that all prerequisites are met in order to use the Kaggle API.
    """
    home = Path.home()
    kaggle_dir = home / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_dir.exists():
        logger.info("Creating ~/.kaggle directory")
        kaggle_dir.mkdir()

    if not kaggle_json.exists():
        logger.error(
            "You need to authenticate with Kaggle first in order to download the data."
            " Please follow the instructions here: https://www.kaggle.com/docs/api"
        )
        # Prompt for username and key
        username = input("Enter your Kaggle username: ")
        key = input("Enter your Kaggle key: ")
        kaggle_json.write_text(f'{{"username":"{username}","key":"{key}"}}')
        logger.info("Kaggle API credentials saved to ~/.kaggle/kaggle.json")


def download_data(dataset_url: str, dataset_path: Path) -> None:
    """
    Downloads the dataset from Kaggle and renames it to the expected file name.
    """
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return

    logger.info("Downloading data from Kaggle")
    try:
        api.dataset_download_files(dataset_url, path=dataset_path.parent, unzip=True)
        logger.info(f"Data downloaded and extracted to {dataset_path.parent}")

        # Check if the expected file is present
        expected_file = dataset_path.parent / dataset_path.name
        if not expected_file.exists():
            # Attempt to find and rename the downloaded file
            extracted_files = list(dataset_path.parent.glob("*.csv"))
            if extracted_files:
                downloaded_file = extracted_files[0]
                downloaded_file.rename(expected_file)
                logger.info(f"Renamed {downloaded_file} to {expected_file}")
            else:
                logger.warning(
                    f"Expected dataset file {dataset_path} not found among extracted files"
                )
        else:
            logger.info(f"Expected dataset file: {expected_file}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")


def ensure_dataset_exists(dataset_path: Path, dataset_url: str) -> None:
    """
    Asserts that the dataset exists.
    If not, it downloads the dataset.
    """
    if dataset_path.exists():
        logger.info(f"Dataset already exists at {dataset_path}")
        return

    logger.warning("Dataset not found")
    assert_prerequisites()
    download_data(dataset_url, dataset_path)
