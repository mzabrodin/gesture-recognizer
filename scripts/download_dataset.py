import os
import concurrent.futures
from huggingface_hub import HfApi, hf_hub_download, RepoFolder, RepoFile
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv() # To load HF_TOKEN from .env file if it exists

REPO_ID = "neilrigaud/hagrid-subset"
REPO_TYPE = "dataset"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "data", "raw"))
LIMIT_PER_CLASS = 2500
MAX_WORKERS = 8
TARGET_SPLIT = "train"


def download_file(args):
    file_path, local_dir = args
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=file_path,
            local_dir=local_dir
        )

        return True
    except Exception as e:
        print(f"Failed to download {file_path}: {e}")
        return False


def main():
    api = HfApi()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Initializing dataset download: {REPO_ID} (Split: {TARGET_SPLIT})")

    try:
        tree = api.list_repo_tree(repo_id=REPO_ID, repo_type=REPO_TYPE, path_in_repo=TARGET_SPLIT, recursive=False)
        classes = [
            item for item in tree
            if isinstance(item, RepoFolder)
        ]
    except Exception as e:
        print(f"Error retrieving repository structure: {e}")
        return

    total_downloaded = 0

    for class_folder in classes:
        class_name = class_folder.path.split("/")[-1]
        print(f"Processing class: {class_name}")

        try:
            files = api.list_repo_tree(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                path_in_repo=class_folder.path,
                recursive=False
            )
        except Exception as e:
            print(f"Error accessing class {class_name}: {e}")
            continue

        image_files = [
            f.path for f in files
            if isinstance(f, RepoFile) and f.path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]

        files_to_download = image_files[:LIMIT_PER_CLASS]

        if not files_to_download:
            continue

        tasks = [(f, OUTPUT_DIR) for f in files_to_download]

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(executor.map(download_file, tasks), total=len(tasks), desc=class_name))

        total_downloaded += sum(results)

    print(f"Download completed. Total files saved: {total_downloaded}")


if __name__ == "__main__":
    main()
