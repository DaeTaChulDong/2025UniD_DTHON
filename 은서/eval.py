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
# 💡 [핵심] code444.py에 정의된 최종 클래스와 함수만 가져옵니다.
from code444 import (
    seed_everything,
    find_jsons,
    UniDSetVLM,      # 💡 code444.py의 데이터셋 클래스
    collate_fn_vlm,  # 💡 code444.py의 collate 함수
    BestVLM,         # 💡 code444.py의 모델 클래스
    CFG,
    box_cxcywh_to_xyxy,
    iou_xywh_pixel   # 💡 mIoU 계산 함수
)

# 💡 [추가] CLIPProcessor는 모델 로드 시 필요
from transformers import CLIPProcessor 

# --- [2. 모델 로드 Helper (BestVLM 맞춤)] ---
# 💡 [수정] Vocab을 사용하지 않고 CLIP 설정으로 모델을 복원합니다.
def _load_model_from_ckpt(ckpt_path: str, device: torch.device):
    """ 저장된 체크포인트(.pth)에서 모델과 설정을 불러옵니다. """
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # 1. 필요한 설정값 추출
    clip_model_name = ckpt.get("clip_model_name", CFG.CLIP_MODEL_NAME)
    dim = ckpt.get("dim", CFG.DIM)
    img_size = ckpt.get("img_size", CFG.IMG_SIZE)
    no_pretrain = ckpt.get("no_pretrain", False)

    # 2. 모델 인스턴스화 (BestVLM은 vocab_size가 필요 없음)
    model = BestVLM(clip_model_name=clip_model_name,
                         dim=dim,
                         pretrained_backbone=not no_pretrain,
                         img_size=img_size).to(device)
    
    # 3. CLIP Processor 로드 (데이터 로딩 시 필요)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    model.load_state_dict(ckpt["model_state"])
    model.eval() 
    
    print(f"모델 로드 완료: {ckpt_path}")
    # 💡 [수정] Vocab 대신 clip_processor 반환
    return model, clip_processor, img_size

# --- [3. mIoU 계산을 위한 평가 루프] ---
def evaluate_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 체크포인트에서 모델과 CLIP Processor 로드
    model, clip_processor, img_size = _load_model_from_ckpt(args.ckpt, device)

    # 2. '검증용(valid)' 데이터 로더 구성
    json_files = find_jsons(args.json_dir)
    # 💡 [핵심] UniDSetVLM에 CLIP Processor를 전달
    valid_ds = UniDSetVLM(json_files, args.jpg_dir, clip_processor=clip_processor,
                       img_size=img_size)
    
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_fn_vlm)

    all_ious = [] 
    
    with torch.no_grad(): 
        loop = tqdm(valid_dl, desc="Evaluating", leave=True)
        
        for pixel_values, input_ids, attention_mask, targets, meta in loop:
            # 💡 [추가] collate_fn이 빈 배치를 반환할 수 있음
            if pixel_values is None: continue 
            
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # 3. 모델 추론
            pred = model(pixel_values, input_ids, attention_mask) 

            # 4. 배치 내 각 샘플에 대해 IoU 계산
            for i in range(targets.size(0)):
                if targets[i] is not None:
                    W, H = meta[i]["meta_orig_size"] 
                    
                    # (1) 예측 BBox (Normalized -> Pixel)
                    cx, cy, nw, nh = [float(v) for v in pred[i].cpu().numpy().tolist()]
                    pred_x = (cx - nw / 2.0) * W; pred_y = (cy - nh / 2.0) * H
                    pred_w = nw * W; pred_h = nh * H
                    
                    # (2) 정답 BBox (Normalized -> Pixel)
                    gt = [float(v) for v in targets[i].numpy().tolist()]
                    gt_x = (gt[0] - gt[2] / 2.0) * W; gt_y = (gt[1] - gt[3] / 2.0) * H
                    gt_w = gt[2] * W; gt_h = gt[3] * H
                    
                    # (3) mIoU 계산 (code444.py에서 import)
                    iou = iou_xywh_pixel([pred_x, pred_y, pred_w, pred_h], [gt_x, gt_y, gt_w, gt_h])
                    all_ious.append(iou)

    # 5. 최종 mIoU (평균) 계산 및 출력
    if all_ious:
        mIoU = float(np.mean(all_ious))
        print("=======================================")
        print(f"✅ [평가 완료] mIoU: {mIoU:.6f}")
        print("=======================================")
    else:
        print(f"경고: 평가할 BBox 정답이 하나도 없습니다. {args.json_dir} 경로를 확인하세요.")

# --- [4. CLI] ---
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", type=str, required=True, help="평가용 JSON 파일 디렉토리")
    ap.add_argument("--jpg_dir", type=str, required=True, help="평가용 JPG 이미지 디렉토리")
    ap.add_argument("--ckpt", type=str, required=True, help="학습된 .pth 체크포인트 경로")
    
    ap.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE * 2) 
    ap.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)
    return ap.parse_args()

def main():
    seed_everything(CFG.SEED)
    args = get_args()
    evaluate_loop(args)

if __name__ == "__main__":
    main()
