import os
import argparse
from huggingface_hub import hf_hub_download


def download_model(model_name: str, target_dir: str):
    """
    Downloads a quantized GGUF model for the Electricity Theft Detection system.
    Defaults to Phi-3-mini-4k-instruct-q4 for edge-compatible inference.
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Mapping common names to specific HuggingFace GGUF repos
    model_map = {
        "phi-3-mini": {
            "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
            "filename": "Phi-3-mini-4k-instruct-q4.gguf"
        }
    }

    if model_name not in model_map:
        print(f"Model {model_name} not supported. Defaulting to phi-3-mini.")
        model_name = "phi-3-mini"

    config = model_map[model_name]

    print(f"Downloading {config['filename']} from {config['repo_id']}...")

    try:
        path = hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['filename'],
            local_dir=target_dir
        )
        print(f"Successfully downloaded model to: {path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download quantized SLM for edge inference.")
    parser.add_argument("--model", type=str, default="phi-3-mini", help="Model name to download")
    parser.add_argument("--output", type=str, default="models/", help="Output directory")

    args = parser.parse_args()
    download_model(args.model, args.output)