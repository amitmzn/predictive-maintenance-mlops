
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

repo_id = "amitmzn/Predictive-Maintenance"
repo_type = "dataset"

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise SystemExit("HF_TOKEN is not set. Add it to Colab env or GitHub Actions secrets.")

api = HfApi(token=hf_token)

# Ensure dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f" Dataset repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f" Dataset '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f" Dataset repository '{repo_id}' created.")

# Resolve paths relative to this script
repo_root = Path(__file__).resolve().parents[1]

candidate_data_dirs = [
    repo_root / "data",
    repo_root / "predictive_maintenance_project" / "data",
]
candidate_csv_files = [
    repo_root / "data" / "engine_data.csv",
    repo_root / "predictive_maintenance_project" / "data" / "engine_data.csv",
    repo_root / "engine_data.csv",
]

data_dir = next((p for p in candidate_data_dirs if p.exists() and p.is_dir()), None)
csv_file = next((p for p in candidate_csv_files if p.exists() and p.is_file()), None)

if data_dir and any(child.is_file() for child in data_dir.iterdir()):
    files_in_folder = [p for p in data_dir.iterdir() if p.is_file()]
    print(f" Uploading {len(files_in_folder)} file(s) from {data_dir.as_posix()}...")
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="",
    )
    print(" Data uploaded successfully!")
elif csv_file:
    print(f" Uploading file {csv_file.as_posix()}...")
    api.upload_file(
        path_or_fileobj=str(csv_file),
        path_in_repo="engine_data.csv",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(" Data uploaded successfully!")
else:
    print(" No local data folder/file found. Skipping upload.")
    print("  (Normal in CI if HF dataset is already populated.)")

print(f"\\n Done. Dataset: https://huggingface.co/datasets/{repo_id}")
