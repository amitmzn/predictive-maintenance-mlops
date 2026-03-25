from huggingface_hub import HfApi
from pathlib import Path
import os

api = HfApi(token=os.getenv("HF_TOKEN"))


script_dir = Path(__file__).resolve().parent
deployment_folder = script_dir.parent / "deployment"

api.upload_folder(
    folder_path=str(deployment_folder),
    repo_id="amitmzn/predictive-maintenance-app",
    repo_type="space",
    path_in_repo="",
)
print("Deployment files uploaded to HF Space!")
