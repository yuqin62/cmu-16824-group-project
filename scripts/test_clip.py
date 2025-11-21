import os
from src.models.clip_backbone import load_clip

def main():
    # Set HuggingFace cache directory
    os.environ["HF_HOME"] = "checkpoints/clip"

    # Load CLIP model (downloads into checkpoints/clip/ automatically)
    model, processor = load_clip(device="cpu")

    print("CLIP model and processor loaded successfully.")

if __name__ == "__main__":
    main()

