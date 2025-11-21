import os
import sys

# ------------------------------------------------------------------
# Ensure project root is on sys.path so "src" can be imported
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Optional: force HuggingFace cache into checkpoints/clip
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "checkpoints", "clip")

from src.models.clip_backbone import load_clip


def main():
    # Use "cpu" for safety; teammates with GPU can switch to "cuda"
    model, processor = load_clip(device="cpu")
    print("CLIP model and processor loaded successfully.")


if __name__ == "__main__":
    main()
