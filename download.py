from huggingface_hub import snapshot_download
model_id="lmms-lab/llava-onevision-qwen2-7b-ov"
snapshot_download(repo_id=model_id, local_dir="llava-ov-7b-ov",
                  local_dir_use_symlinks=False, revision="main")