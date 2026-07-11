# ViT-LPG-Phish for Food-101

A Vision Transformer from scratch with LocalPhishGate + Phish activation trained on Food-101.

## Key Features
- **Phish Activation**: x * erf(x/sqrt2) * sigmoid(x) - smoother than GELU
- **LPG Block**: Local depthwise conv + gate for better local features in ViT
- **Trained from scratch**: No ImageNet pretraining
- **Result**: 62.74% Top-1 on Food-101 in 25 epochs

## Quickstart
1. `pip install torch torchvision tqdm`
2. `python Train model.py` # auto downloads Food-101 and trains the vit from scratch
3. `python predict.py --img your_food.jpg`
