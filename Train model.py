"""
ViT + LPG + Phish on Food-101
Auto-download dataset, train 25 epochs, save best model
"""

import math
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_CLASSES = 101
IMG_SIZE = 224
PATCH_SIZE = 16
DIM = 384
DEPTH = 8
HEADS = 6
MLP_RATIO = 4
EPOCHS = 25
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.05
NUM_WORKERS = min(8, os.cpu_count() or 4)
DATA_ROOT = "./data"
CKPT_PATH = "best_vit_lpg_phish_v2.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Phish Activation
# ---------------------------------------------------------------------------
class Phish(nn.Module):
    """Phish(x) = x * erf(x / sqrt(2)) * sigmoid(x)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.erf(x / math.sqrt(2.0)) * torch.sigmoid(x)

# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # (B, D, H/P, W/P)
        x = x.flatten(2) # (B, D, N)
        x = x.transpose(1, 2) # (B, N, D)
        return x

# ---------------------------------------------------------------------------
# MLP block
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, act_layer=None):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        act = act_layer if act_layer is not None else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            act,
            nn.Linear(hidden, dim),
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# LocalPhishGate (LPG)
# ---------------------------------------------------------------------------
class LocalPhishGate(nn.Module):
    def __init__(self, dim, num_patches, kernel_size=7):
        super().__init__()
        self.dim = dim
        h = w = int(math.sqrt(num_patches))
        assert h * w == num_patches
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

# ---------------------------------------------------------------------------
# Transformer Block with LPG + Phish
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# ViT + LPG + Phish
# ---------------------------------------------------------------------------
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
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None: nn.init.zeros_(m.bias)

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

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def gpu_mem_mb():
    return torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0

def get_dataloaders(root=DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    print("Downloading Food-101 if not present...")
    train_ds = datasets.Food101(root=root, split="train", transform=train_tf, download=True)
    val_ds = datasets.Food101(root=root, split="test", transform=val_tf, download=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def train_one_epoch(model, loader, optimizer, scaler, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Train E{epoch:02d}", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=DEVICE.type):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        correct += logits.argmax(1).eq(labels).sum().item()
        total += bs
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100*correct/total:.2f}%")
    return 100.0 * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, epoch):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Val E{epoch:02d}", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type):
            logits = model(imgs)
            loss = criterion(logits, labels)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        correct += logits.argmax(1).eq(labels).sum().item()
        total += bs
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100*correct/total:.2f}%")
    return 100.0 * correct / total, total_loss / total

def main():
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Training ViT+LPG+Phish for {EPOCHS} epochs")
    print("="*60)

    model = ViTLPGPhish(img_size=IMG_SIZE, patch_size=PATCH_SIZE, num_classes=NUM_CLASSES,
                        embed_dim=DIM, depth=DEPTH, num_heads=HEADS, mlp_ratio=MLP_RATIO).to(DEVICE)
    print(f"Params: {count_params(model):,}")

    train_loader, val_loader = get_dataloaders()
    print(f"Train: {len(train_loader.dataset):,} Val: {len(val_loader.dataset):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, epoch)
        val_acc, val_loss = evaluate(model, val_loader, criterion, epoch)
        scheduler.step()
        epoch_time = time.time() - start

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "best_acc": best_acc}, CKPT_PATH)
            print(f" >> Saved new best: {best_acc:.2f}%")

        mem_str = f"GPU: {gpu_mem_mb():.0f}MB" if torch.cuda.is_available() else ""
        print(f"Epoch {epoch:02d}/{EPOCHS} | train={train_acc:.2f}% | val={val_acc:.2f}% | loss={val_loss:.4f} | {epoch_time:.1f}s | best={best_acc:.2f}% {mem_str}")

    print("="*60)
    print(f"Training done! Best Val Acc: {best_acc:.2f}%")
    print(f"Model saved to: {CKPT_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()