from transformers import CLIPModel, CLIPProcessor

def load_clip(model_name="openai/clip-vit-base-patch32", device="cuda"):
    """
    Load a CLIP model from HuggingFace. 
    The actual weights are not stored in the repo.
    """
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    return model, processor
