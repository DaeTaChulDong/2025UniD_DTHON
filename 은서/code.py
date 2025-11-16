import os
import json
import math
import random
import argparse
import numpy as np
from glob import glob
from typing import List, Tuple, Dict, Any
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import generalized_box_iou_loss
from tqdm import tqdm
import torchvision.transforms as T 

from transformers import CLIPProcessor, CLIPTextModel, CLIPConfig 

# torchvision 백본 체크
try:
    from torchvision.models import resnet18, ResNet18_Weights
    _BACKBONE_OK = True
except Exception:
    _BACKBONE_OK = False

# --- [1. CFG (Global Settings) ] ---
class CFG:
    IMG_SIZE: int = 512
    EPOCHS: int = 20 
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 16 
    SEED: int = 42
    DIM: int = 256
    NUM_WORKERS: int = 4
    NO_PRETRAIN: bool = False
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    
    # 💡 [Elice 서버 절대 경로] 
    TRAIN_JSON_DIR: str = "/home/elicer/data/train_valid/train"
    TRAIN_JPG_DIR: str = "/home/elicer/data/train_valid/train"
    VALID_JSON_DIR: str = "/home/elicer/data/train_valid/valid"
    VALID_JPG_DIR: str = "/home/elicer/data/train_valid/valid"
    
    CKPT_PATH: str = "./outputs/ckpt/best_miou_model.pth"

# --- [2. HELPER FUNCTIONS (순서대로 정의)] ---
def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_json(path: str) -> Dict[str, Any]:
    """ JSONDecodeError 해결을 위해 utf-8-sig 인코딩 사용 """
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        return None

def find_jsons(json_dir: str) -> List[str]:
    """ find_jsons가 UniDSetVLM보다 먼저 정의되어야 함 """
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"json_dir not found: {json_dir}")
    json_files = sorted(glob(os.path.join(json_dir, "**", "*.json"), recursive=True))
    if json_files: return json_files
    print(f"경고: {json_dir} 에서 .json 파일을 찾을 수 없습니다.")
    return []

def get_image_path(json_path: str, data: Dict[str, Any], jpg_dir: str) -> str:
    """ get_image_path가 UniDSetVLM보다 먼저 정의되어야 함 """
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg", None)
    json_dir_name = os.path.basename(os.path.dirname(json_path)) 
    
    jpg_folder = os.path.dirname(json_path).replace(json_dir_name, json_dir_name.replace("_json", "_jpg"))

    if jpg_name:
        path = os.path.join(jpg_folder, jpg_name)
        if os.path.exists(path):
            return path
            
    base = os.path.splitext(os.path.basename(json_path))[0]
    jpg_name_fallback = base.replace("MI3", "MI2") + ".jpg" 
    path_fallback = os.path.join(jpg_folder, jpg_name_fallback)
    if os.path.exists(path_fallback):
        return path_fallback

    raise FileNotFoundError(f"JPG ({jpg_name} or {jpg_name_fallback}) 못찾음: {json_path}")

def is_visual_ann(a: dict) -> bool:
    """ is_visual_ann가 UniDSetVLM보다 먼저 정의되어야 함 """
    cid = str(a.get("class_id", "") or "")
    has_q = bool(str(a.get("visual_instruction", "") or "").strip())
    looks_visual = cid.startswith("V")
    has_bbox = "bounding_box" in a
    
    return looks_visual and has_q and has_bbox

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def simple_tokenize(s: str) -> List[str]:
    """베이스라인의 간단한 토크나이저"""
    s = (s or "")
    s = s.replace("##", " ").replace(",", " ").replace("(", " ").replace(")", " ")
    s = s.replace(":", " ").replace("?", " ").replace("!", " ").replace("·", " ")
    return [t for t in s.strip().split() if t]

class Vocab:
    """베이스라인의 Vocab 클래스 (ImportError 해결용)"""
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.freq: Dict[str, int] = {}
        self.itos: List[str] = ["<pad>", "<unk>"] # 0=pad, 1=unk
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    def build(self, texts: List[str]):
        for s in texts:
            for tok in simple_tokenize(s): 
                self.freq[tok] = self.freq.get(tok, 0) + 1
        for tok, f in sorted(self.freq.items(), key=lambda x: (-x[1], x[0])):
            if f >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
    
    def encode(self, s: str, max_len: int = 40) -> List[int]:
        toks = simple_tokenize(s)[:max_len]
        if not toks:
            return [1]
        return [self.stoi.get(t, 1) for t in toks]
    
# --- [3. MODEL CLASSES (하위 → 상위)] ---
# (ImageEncoder, CLIPTextEncoder, CrossAttentionBBox, BestVLM 클래스는 이전과 동일)
# ... (생략: 코드가 길어 여기서는 이전과 동일하다고 가정하고, 파일에 포함시킵니다.) ...
class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = CFG.DIM, pretrained: bool = True):
        super().__init__()
        if not _BACKBONE_OK:
            raise ImportError("torchvision.models.resnet18을 찾을 수 없습니다.")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=weights)
        layers = list(m.children())[:-2]
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Conv2d(512, out_dim, 1) 

    def forward(self, x):
        f = self.backbone(x)
        f = self.proj(f)
        return f 

class CLIPTextEncoder(nn.Module):
    def __init__(self, clip_model_name: str, out_dim: int = CFG.DIM):
        super().__init__()
        self.clip = CLIPTextModel.from_pretrained(clip_model_name)
        clip_dim = self.clip.config.hidden_size
        self.proj = nn.Linear(clip_dim, out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.clip(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        q = self.proj(pooled_output)
        return q 

class CrossAttentionBBox(nn.Module):
    def __init__(self, dim: int = CFG.DIM):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, 4)
        )
    def forward(self, q_vec: torch.Tensor, fmap: torch.Tensor) -> torch.Tensor:
        B, D, H, W = fmap.shape
        q = self.q_proj(q_vec)
        K = self.k_proj(fmap)
        V = self.v_proj(fmap)
        Kf = K.flatten(2).transpose(1, 2)
        Vf = V.flatten(2).transpose(1, 2)
        q = q.unsqueeze(1)
        attn = torch.matmul(q, Kf.transpose(1, 2)) / math.sqrt(D)
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, Vf).squeeze(1)
        pred = self.bbox_head(ctx)
        pred = torch.sigmoid(pred)
        return pred

class BestVLM(nn.Module):
    def __init__(self, clip_model_name: str, dim: int = CFG.DIM, pretrained_backbone: bool = True, img_size: int = CFG.IMG_SIZE):
        super().__init__()
        self.txt = CLIPTextEncoder(clip_model_name=clip_model_name, out_dim=dim)
        self.img = ImageEncoder(out_dim=dim, pretrained=pretrained_backbone)
        self.head = CrossAttentionBBox(dim=dim)

    def forward(self, pixel_values: torch.Tensor, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        
        q = self.txt(input_ids, attention_mask)
        fmap = self.img(pixel_values)
        pred_norm = self.head(q, fmap)
        return pred_norm
    
# --- [4. DATASET CLASSES] ---
class UniDSetVLM(Dataset):
    def __init__(self, json_files: List[str], jpg_dir: str, 
                 clip_processor: CLIPProcessor, 
                 max_txt_len: int = 77, img_size: int = 512):
        
        self.items = []
        self.processor = clip_processor
        self.max_txt_len = max_txt_len
        
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])
        
        print("데이터셋 로드 중... (JSON 파일 처리 중)")
        for jf in tqdm(json_files, desc="JSON 파일 로드", unit="file"):
            data = read_json(jf) 
            if not data: continue
            
            try:
                img_path = get_image_path(jf, data, jpg_dir=jpg_dir) 
            except FileNotFoundError:
                continue

            ann = data.get("learning_data_info", {}).get("annotation", [])
            for a in ann:
                if not is_visual_ann(a): 
                    continue
                
                self.items.append({
                    "img": img_path,
                    "query": str(a.get("visual_instruction", "")).strip(),
                    "bbox": a.get("bounding_box", None),
                    "query_id": a.get("instance_id", ""),
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        
        try:
            img = Image.open(it["img"]).convert("RGB") 
        except Exception as e:
            return None 

        W, H = img.size
        
        qtxt = it["query"]
        txt_encoding = self.processor(
            text=qtxt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len
        )
        
        img_t = self.img_transform(img)

        target = None
        if it["bbox"] is not None:
            x, y, w, h = it["bbox"]
            # 💡 [핵심] BBox 정규화
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            nw = w / W
            nh = h / H
            target = torch.tensor([cx, cy, nw, nh], dtype=torch.float32)
        
        if target is None:
             return None 

        sample = {
            "input_ids": txt_encoding["input_ids"].squeeze(0),
            "attention_mask": txt_encoding["attention_mask"].squeeze(0),
            "pixel_values": img_t, 
            "target": target,
            "meta_query_id": it["query_id"],
            "meta_orig_size": (W, H),
        }
        return sample

# --- [code444.py - collate_fn_vlm 함수 교체] ---

def collate_fn_vlm(batch: List[Dict[str, Any]]):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, None, None

    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    
    targets = torch.stack([b["target"] for b in batch]) 
    
    # 💡 [핵심 수정] 키 이름 통일: "meta_orig_size"를 사용합니다.
    meta = [
        {
            "query_id": b["meta_query_id"], 
            "meta_orig_size": b["meta_orig_size"] # <-- 키 이름 수정
        }
        for b in batch
    ]
    return pixel_values, input_ids, attention_mask, targets, meta

# --- [mIoU 계산을 위한 Helper] ---
def iou_xywh_pixel(pred_xywh, gt_xywh):
    """ BBox 일치도 (IoU)를 픽셀 기반으로 계산합니다. """
    px, py, pw, ph = pred_xywh
    gx, gy, gw, gh = gt_xywh
    px2, py2 = px + pw, py + ph
    gx2, gy2 = gx + gw, gy + gh
    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = pw * ph + gw * gh - inter if (pw * ph + gw * gh - inter) > 0 else 1e-6
    return inter / union

def calculate_miou_on_loader(model, data_loader, device):
    """ 현재 모델의 mIoU를 계산하여 반환합니다. """
    model.eval()
    all_ious = []
    
    with torch.no_grad():
        eval_loop = tqdm(data_loader, desc="Calculating mIoU", leave=False, unit="batch")
        
        for pixel_values, input_ids, attention_mask, targets, meta in eval_loop:
            if pixel_values is None: continue 
            
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            pred_norm = model(pixel_values, input_ids, attention_mask)
            
            for i in range(pred_norm.size(0)):
                if targets[i] is not None:
                    W, H = meta[i]["meta_orig_size"] 
                    
                    # (1) 예측 BBox (Normalized -> Pixel)
                    cx, cy, nw, nh = [float(v) for v in pred_norm[i].cpu().numpy().tolist()]
                    pred_x = (cx - nw / 2.0) * W; pred_y = (cy - nh / 2.0) * H
                    pred_w = nw * W; pred_h = nh * H
                    
                    # (2) 정답 BBox (Normalized -> Pixel)
                    gt_norm = targets[i].numpy().tolist()
                    gt_x = (gt_norm[0] - gt_norm[2] / 2.0) * W; gt_y = (gt_norm[1] - gt_norm[3] / 2.0) * H
                    gt_w = gt_norm[2] * W; gt_h = gt_norm[3] * H
                    
                    iou = iou_xywh_pixel([pred_x, pred_y, pred_w, pred_h], [gt_x, gt_y, gt_w, gt_h])
                    all_ious.append(iou)

    model.train() # 학습 모드로 복귀
    return float(np.mean(all_ious)) if all_ious else 0.0

# --- [5. TRAIN LOOP & CLI] ---
def train_loop(args):
    # 💡 [추가] 서브셋 크기 설정 (전체 데이터 80000의 1/100)
    SUBSET_DIVISOR = 10 # 800개 샘플

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. CLIP Processor 로드
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    
    # 2. 데이터셋 생성
    train_json_files = find_jsons(args.train_json_dir)
    full_train_ds = UniDSetVLM(train_json_files, args.train_jpg_dir, clip_processor, img_size=args.img_size)

    # --- 💡 [핵심: 데이터셋 1/160 샘플링 로직] 💡 ---
    full_size = len(full_train_ds)
    subset_size = full_size // SUBSET_DIVISOR
    indices = torch.randperm(full_size)[:subset_size] 
    train_ds = torch.utils.data.Subset(full_train_ds, indices)
    # ----------------------------------------------------

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_fn_vlm)

    model = BestVLM(clip_model_name=args.clip_model_name,
                    dim=args.dim, 
                    pretrained_backbone=not args.no_pretrain,
                    img_size=args.img_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    total_samples = len(train_ds)
    print(f"Total {total_samples} (질의, BBox) 쌍 샘플을 사용합니다 (전체의 1/{SUBSET_DIVISOR}).")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs} (Train)", leave=False)
        
        for pixel_values, input_ids, attention_mask, targets, meta in loop:
            if pixel_values is None: continue 
            
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            t = targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred = model(pixel_values, input_ids, attention_mask)
                
                # 💡 [mIoU 80% 전략] L1 + GIoU Loss 
                loss_l1 = F.smooth_l1_loss(pred, t, reduction="mean")
                pred_xyxy = box_cxcywh_to_xyxy(pred)
                t_xyxy = box_cxcywh_to_xyxy(t)
                loss_giou = generalized_box_iou_loss(pred_xyxy, t_xyxy).mean()
                loss = loss_l1 + loss_giou 

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            batch_loss = float(loss.item())
            running_loss += batch_loss * pixel_values.size(0)
            loop.set_postfix(loss=batch_loss)
            
        scheduler.step()
        avg_loss = running_loss / total_samples if total_samples > 0 else 0
        
        # 💡 [핵심] Epoch 종료 후 mIoU 계산
        current_miou = calculate_miou_on_loader(model, train_dl, device)

        print(f"Epoch {epoch} Summary: Avg_loss={avg_loss:.4f} | mIoU={current_miou:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")

    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "clip_model_name": args.clip_model_name,
        "dim": args.dim,
        "no_pretrain": args.no_pretrain,
        "img_size": args.img_size,
    }, args.save_ckpt)
    print(f"[Saved] {args.save_ckpt}")

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json_dir", type=str, default=CFG.TRAIN_JSON_DIR)
    ap.add_argument("--train_jpg_dir", type=str, default=CFG.TRAIN_JPG_DIR)
    ap.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    ap.add_argument("--img_size", type=int, default=CFG.IMG_SIZE)
    ap.add_argument("--dim", type=int, default=CFG.DIM)
    ap.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)
    ap.add_argument("--clip_model_name", type=str, default=CFG.CLIP_MODEL_NAME)
    ap.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    ap.add_argument("--lr", type=float, default=CFG.LEARNING_RATE)
    ap.add_argument("--no_pretrain", action="store_true")
    ap.add_argument("--save_ckpt", type=str, default=CFG.CKPT_PATH)
    return ap.parse_args()

def main():
    seed_everything(CFG.SEED)
    args = get_args()
    train_loop(args)

if __name__ == "__main__":
    main()
