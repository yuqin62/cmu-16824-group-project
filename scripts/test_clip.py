from src.models.clip_backbone import load_clip

def main():
    model, processor = load_clip(device="cpu")  # or "cuda" if they have GPU
    print("CLIP model and processor loaded successfully.")

if __name__ == "__main__":
    main()
