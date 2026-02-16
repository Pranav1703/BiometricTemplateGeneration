import os
import re
from pathlib import Path
from src.utils.logger import get_logger

try:
    import gdown

    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    print("⚠️  gdown not available. Install with: pip install gdown")

logger = get_logger("model_downloader")


def is_valid_model_file(filename: str) -> bool:
    """Check if filename matches the pattern *_arcface_model.pth"""
    return filename.endswith("_arcface_model.pth")


def download_drive_folder(folder_url: str, output_dir: str) -> bool:
    """
    Download all valid model files from Google Drive folder.

    Args:
        folder_url: Google Drive folder URL
        output_dir: Local directory to save models

    Returns:
        bool: True if download was successful, False otherwise
    """
    if not GDOWN_AVAILABLE:
        logger.error("gdown not available. Install with: pip install gdown")
        return False

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Downloading models from Google Drive folder to {output_dir}")
        logger.info(f"Folder URL: {folder_url}")

        # Extract folder ID from URL
        # Handle both formats: https://drive.google.com/drive/folders/ID and https://drive.google.com/drive/folders/ID?usp=sharing
        folder_id_match = re.search(r"/folders/([a-zA-Z0-9_-]+)", folder_url)
        if not folder_id_match:
            logger.error("Invalid Google Drive folder URL format")
            return False

        folder_id = folder_id_match.group(1)

        # Use gdown to download folder
        gdown.download_folder(
            id=folder_id, output=output_dir, quiet=False, use_cookies=False
        )

        # Verify downloaded files
        downloaded_files = []
        for file in os.listdir(output_dir):
            if is_valid_model_file(file):
                downloaded_files.append(file)
                logger.info(f"Downloaded model: {file}")

        if not downloaded_files:
            logger.warning(
                "No valid model files (*_arcface_model.pth) found in the downloaded content"
            )
            return False

        logger.info(f"Successfully downloaded {len(downloaded_files)} model files")
        return True

    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        return False


def check_and_download_models(models_dir: str, drive_folder_url: str = "") -> bool:
    """
    Check if models exist in directory, download from Drive if not.

    Args:
        models_dir: Directory containing model files
        drive_folder_url: Google Drive folder URL (optional, defaults to main models folder)

    Returns:
        bool: True if models are available (either existed or downloaded successfully)
    """
    # Use default URL if none provided
    if not drive_folder_url:
        drive_folder_url = "https://drive.google.com/drive/folders/1hh4CHY4jFk8gJhsPziOKqKNlbuW4nD08?usp=sharing"

    # Check if directory exists
    if not os.path.exists(models_dir):
        logger.info(f"Models directory {models_dir} does not exist")
        return download_drive_folder(drive_folder_url, models_dir)

    # Check for existing model files
    existing_models = []
    for file in os.listdir(models_dir):
        if is_valid_model_file(file):
            existing_models.append(file)

    if existing_models:
        logger.info(
            f"Found {len(existing_models)} existing model(s): {existing_models}"
        )
        return True
    else:
        logger.info(f"No valid models found in {models_dir}. Downloading...")
        return download_drive_folder(drive_folder_url, models_dir)


def main():
    """Main function to download models"""
    from src.config import SAVED_MODELS_DIR

    success = check_and_download_models(SAVED_MODELS_DIR)
    if success:
        print("[SUCCESS] Models are ready for use!")
    else:
        print("[ERROR] Failed to download models")


if __name__ == "__main__":
    main()
