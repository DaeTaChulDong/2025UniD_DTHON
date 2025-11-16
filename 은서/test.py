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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import torchvision.transforms as T 

# --- [1. code444.py에서 필요한 요소 import] ---
from code444 import (
    seed_everything,
    BestVLM,        # 💡 code444.py의 모델 클래스
    CFG,
    box_cxcywh_to_xyxy,
    read_json       
)
from transformers import CLIPProcessor 

# --- [2. 모델 로드 Helper (code444.py 호환)] ---
def _load_model_from_ckpt(ckpt_path: str, device: torch.device):
    """ 저장된 체크포인트(.pth)에서 모델과 설정을 불러옵니다. """
    ckpt = torch.load(ckpt_path, map_location=device)
    
    clip_model_name = ckpt.get("clip_model_name", CFG.CLIP_MODEL_NAME)
    dim = ckpt.get("dim", CFG.DIM)
    img_size = ckpt.get("img_size", CFG.IMG_SIZE)
    no_pretrain = ckpt.get("no_pretrain", False)

    model = BestVLM(clip_model_name=clip_model_name,
                         dim=dim,
                         pretrained_backbone=not no_pretrain,
                         img_size=img_size).to(device)
    
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    model.load_state_dict(ckpt["model_state"])
    model.eval() 
    
    print(f"모델 로드 완료: {ckpt_path}")
    return model, clip_processor, img_size

# --- [3. 💡 [핵심] Test 데이터 전용 Dataset 클래스 (CSV 기반) 💡] ---
class TestDSetVLM(Dataset):
    """
    Test 데이터셋 로더 (sample_submission.csv 의 순서를 보장)
    """
    def __init__(self, submission_df: pd.DataFrame, jpg_dir: str, 
                 clip_processor: CLIPProcessor, 
                 max_txt_len: int = 77, img_size: int = 512):
        
        self.items = []
        self.processor = clip_processor
        self.max_txt_len = max_txt_len
        
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])
        
        print("Test 데이터셋 로드 중... (sample_submission.csv  기반)")
        
        for idx, row in tqdm(submission_df.iterrows(), total=len(submission_df), desc="Test 샘플 매칭"):
            query_id = row["query_id"]
            query_text = row["query_text"]
            
            # 1. 💡 [핵심] query_id 로부터 이미지 파일명(MI2...) 추론
            # 예: MI3_240819_TY1_0011_1_V02-8_1 -> MI2_240819_TY1_0011_1.jpg
            parts = query_id.split('_')
            if len(parts) < 5:
                print(f"경고: 예기치 않은 query_id 형식: {query_id}")
                continue
                
            base_name = "_".join(parts[:5]) # 'MI3_240819_TY1_0011_1'
            img_name = base_name.replace("MI3", "MI2") + ".jpg" # 'MI2_...jpg'
            
            img_path = os.path.join(jpg_dir, img_name)

            if not os.path.exists(img_path):
                # print(f"경고: {img_path} 이미지를 찾을 수 없습니다. (Query ID: {query_id})")
                continue
                
            self.items.append({
                "img_path": img_path,
                "query_text": query_text,
                "query_id": query_id,
            })
        
        print(f"📌 [Test] 최종 매칭된 샘플 수: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        
        try:
            img = Image.open(it["img_path"]).convert("RGB") 
        except Exception as e:
            return None 
        W, H = img.size
        
        qtxt = it["query_text"]
        
        txt_encoding = self.processor(
            text=qtxt,
            images=None, 
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len
        )
        
        img_t = self.img_transform(img)

        sample = {
            "input_ids": txt_encoding["input_ids"].squeeze(0),
            "attention_mask": txt_encoding["attention_mask"].squeeze(0),
            "pixel_values": img_t, 
            "query_id": it["query_id"],
            "query_text": it["query_text"],
            "orig_size": (W, H),
        }
        return sample

def collate_fn_test(batch: List[Dict[str, Any]]):
    """ Test 데이터 전용 collate_fn """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, None

    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    
    meta = [
        {
            "query_id": b["query_id"], 
            "query_text": b["query_text"],
            "orig_size": b["orig_size"]
        }
        for b in batch
    ]
    return pixel_values, input_ids, attention_mask, meta

# --- [4. 메인 추론 루프] ---
def predict_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, clip_processor, img_size = _load_model_from_ckpt(args.ckpt, device)

    # 💡 [핵심] sample_submission.csv  로드
    sub_df = pd.read_csv(args.submission_csv)
    
    # 💡 [핵심] DataFrame을 기반으로 TestDSetVLM 생성
    test_ds = TestDSetVLM(sub_df, args.jpg_dir, clip_processor=clip_processor,
                         img_size=img_size)
    
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_fn_test)

    # 💡 [핵심] 예측 결과를 {query_id: prediction} 딕셔너리에 저장
    predictions = {}
    
    with torch.no_grad(): 
        loop = tqdm(test_dl, desc="Generating Predictions", leave=True)
        
        for pixel_values, input_ids, attention_mask, meta in loop:
            if pixel_values is None: continue 
            
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            pred_norm = model(pixel_values, input_ids, attention_mask) 

            for i in range(pred_norm.size(0)):
                W, H = meta[i]["orig_size"] 
                
                cx, cy, nw, nh = [float(v) for v in pred_norm[i].cpu().numpy().tolist()]
                
                pred_x = (cx - nw / 2.0) * W
                pred_y = (cy - nh / 2.0) * H
                pred_w = nw * W
                pred_h = nh * H
                
                # 💡 예측 결과를 딕셔너리에 저장
                predictions[meta[i]["query_id"]] = (pred_x, pred_y, pred_w, pred_h)

    # 💡 [핵심] sample_submission.csv  순서대로 결과 매핑
    final_rows = []
    for idx, row in sub_df.iterrows():
        query_id = row["query_id"]
        pred_coords = predictions.get(query_id, (0, 0, 0, 0)) # 💡 매칭된 예측값, 없으면 (0,0,0,0)
        
        final_rows.append({
            "query_id": query_id,
            "query_text": row["query_text"],
            "pred_x": pred_coords[0],
            "pred_y": pred_coords[1],
            "pred_w": pred_coords[2],
            "pred_h": pred_coords[3],
        })

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.DataFrame(final_rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ [저장 완료] 최종 제출 파일 생성 (순서 보장됨): {args.out_csv}")


# --- [5. CLI] ---
def get_args():
    ap = argparse.ArgumentParser()
    # 💡 [수정] --json_dir 대신 --submission_csv 사용
    ap.add_argument("--submission_csv", type=str, required=True, help="정답 순서가 정의된 sample_submission.csv  경로")
    ap.add_argument("--jpg_dir", type=str, required=True, help="Test JPG 이미지 폴더")
    ap.add_argument("--ckpt", type=str, required=True, help="학습된 .pth 체크포인트 경로")
    
    ap.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE) 
    ap.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)
    ap.add_argument("--out_csv", type=str, default="./outputs/preds/test_pred.csv", help="출력 CSV 파일 경로")
    return ap.parse_args()

def main():
    seed_everything(CFG.SEED)
    args = get_args()
    predict_loop(args)

if __name__ == "__main__":
    main()
