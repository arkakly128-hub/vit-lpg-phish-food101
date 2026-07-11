import math
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import argparse
import datetime
import os

# ========= SAME MODEL DEFINITION AS TRAINING =========
NUM_CLASSES = 101
IMG_SIZE = 224
PATCH_SIZE = 16
DIM = 384
DEPTH = 8
HEADS = 6
MLP_RATIO = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download and put in same folder: https://raw.githubusercontent.com/peterhan91/Food101Dataset/master/meta/classes.txt
with open('classes.txt', 'r', encoding='utf-8') as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

class Phish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.erf(x / math.sqrt(2.0)) * torch.sigmoid(x)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, act_layer=None):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        act = act_layer if act_layer is not None else nn.GELU()
        self.net = nn.Sequential(nn.Linear(dim, hidden), act, nn.Linear(hidden, dim))
    def forward(self, x):
        return self.net(x)

class LocalPhishGate(nn.Module):
    def __init__(self, dim, num_patches, kernel_size=7):
        super().__init__()
        self.dim = dim
        h = w = int(math.sqrt(num_patches))
        self.h, self.w = h, w
        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)
        self.pos_bias = nn.Parameter(torch.zeros(1, dim, h, w))
        self.act = Phish()
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, D = x.shape
        x_2d = x.transpose(1, 2).reshape(B, D, self.h, self.w)
        x_2d = self.dw_conv(x_2d) + self.pos_bias
        x_2d = self.act(x_2d)
        x_seq = x_2d.flatten(2).transpose(1, 2)
        x_seq = self.proj(x_seq)
        return x_seq

class TransformerBlockLPG(nn.Module):
    def __init__(self, dim, num_heads, num_patches, mlp_ratio=4):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.lpg = LocalPhishGate(dim, num_patches)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, act_layer=Phish())
    def forward(self, x):
        cls, patches = x[:, :1, :], x[:, 1:, :]
        patches = patches + self.lpg(self.norm0(patches))
        x = torch.cat([cls, patches], dim=1)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class ViTLPGPhish(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=101,
                 embed_dim=384, depth=8, num_heads=6, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlockLPG(embed_dim, num_heads, num_patches, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

# ========= INFERENCE CODE =========
def load_model(model_path):
    model = ViTLPGPhish(img_size=IMG_SIZE, patch_size=PATCH_SIZE, num_classes=NUM_CLASSES,
                        embed_dim=DIM, depth=DEPTH, num_heads=HEADS, mlp_ratio=MLP_RATIO).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} | Best Val Acc: {ckpt['best_acc']:.2f}%")
    print(f"Model loaded on {DEVICE}")
    return model

def save_result(img_path, top1_class, top1_prob):
    result_line = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {os.path.basename(img_path)} | {top1_class} | {top1_prob:.2f}%\n"
    with open("predictions.txt", "a", encoding="utf-8") as f:
        f.write(result_line)

def save_img_with_label(img_path, label, prob):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    text = f"{label}: {prob:.2f}%"
    draw.text((10, 10), text, fill="red")
    folder, name = os.path.split(img_path)
    out_name = os.path.join(folder, f"pred_{name}")
    img.save(out_name)
    print(f"Labeled image saved as: {out_name}")

def predict(model, img_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)

    top1_class = CLASS_NAMES[top5_idx[0][0]]
    top1_prob = top5_prob[0][0].item() * 100

    print(f"\n=== Results for: {os.path.basename(img_path)} ===")
    print(f"Top 1: {top1_class} - {top1_prob:.2f}%\n")
    print("Top 5:")
    for i in range(5):
        print(f"{i+1}. {CLASS_NAMES[top5_idx[0][i]]}: {top5_prob[0][i].item()*100:.2f}%")

    save_result(img_path, top1_class, top1_prob)
    save_img_with_label(img_path, top1_class, top1_prob)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_vit_lpg_phish_v2.pth')
    parser.add_argument('--img', type=str, required=True)
    args = parser.parse_args()

    model = load_model(args.model)
    predict(model, args.img)