import os
from huggingface_hub import HfApi

def upload_workspace():
    api = HfApi()
    repo_id = os.environ["HF_SPACE_REPO_ID"]
    token = os.environ["HF_TOKEN"]
    ignore_dirs = {".venv", "__pycache__", ".git"}
    ignore_exts = {".pyc", ".pyo"}
    
    print(f"Connecting to Hugging Face: {repo_id}...")
    
    for root, dirs, files in os.walk("."):
        # filter dirs in place
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in ignore_exts:
                continue
                
            local_path = os.path.join(root, file)
            # Make path relative to repo root using forward slashes
            repo_path = os.path.relpath(local_path, ".").replace("\\", "/")
            
            print(f"Uploading [{repo_path}] ... ", end="", flush=True)
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="space",
                    token=token
                )
                print("DONE")
            except Exception as e:
                print(f"FAILED: {e}")

if __name__ == "__main__":
    upload_workspace()
