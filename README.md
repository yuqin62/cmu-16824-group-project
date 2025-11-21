# cmu-16824-group-project

## Pretrained Models
Install required packages
pip install --upgrade transformers huggingface_hub torch pillow

We use the following pretrained VLMs from HuggingFace:

- **CLIP ViT-B/32**  
  Model: `openai/clip-vit-base-patch32`  
  Link: https://huggingface.co/openai/clip-vit-base-patch32

The model is **not stored in this repo**.  
It will be automatically downloaded by the `transformers` library the first time you run:

python scripts/test_clip.py in cmu-16824-group-project folder
