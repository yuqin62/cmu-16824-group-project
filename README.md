## 📦 Installation for models

### 1. Create & activate the conda environment

We recommend using a dedicated environment:

#### conda create -n vla-med python=3.10 -y
#### conda activate vla-med

### 2. Install required Python packages

#### pip install --upgrade transformers huggingface_hub torch pillow

### 3. Download the CLIP pretrained model (automatically)

We provide a test script that loads the CLIP model and verifies the installation.

The model will be downloaded automatically into:

"cmu-16824-group-project/checkpoints/clip/"

Run for installation and test: 

#### python scripts/test_clip.py

Expected output: CLIP model and processor loaded successfully.

If this is your first time running the script, HuggingFace will download the weights into the local checkpoints directory:
"/cmu-16824-group-project/checkpoints/clip/" folder




## ℹ️ Other things good to know:
We use the following pretrained VLMs from HuggingFace:

- **CLIP ViT-B/32**  
  Model: `openai/clip-vit-base-patch32`  
  Link: https://huggingface.co/openai/clip-vit-base-patch32

The model is **not stored in this repo**.  



