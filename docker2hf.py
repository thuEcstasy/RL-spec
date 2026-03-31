import os
from huggingface_hub import login, HfApi

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
login("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")

api = HfApi(endpoint="https://hf-mirror.com")
api.create_repo(repo_id="thuEcstasy1/rl-spec-image", exist_ok=True)
api.upload_file(
    path_or_fileobj="rl-spec_latest.tar",
    path_in_repo="rl-spec_latest.tar",
    repo_id="thuEcstasy1/rl-spec-image",
    repo_type="model",
)