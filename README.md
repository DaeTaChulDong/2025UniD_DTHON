# 2025UniD_DTHON

 '2025 Uni-DTHON 데이터톤 트랙' 문서 내 시각요소(표·차트) 위치 예측을 위한 질의기반 비전-언어 모델 개발

CLIP(Text)과 ResNet50(Image) 융합, 평가 지표인 mIoU를 직접 최적화하는 GIoU Loss를 도입하여 성능을 극대화

1) 텍스트 인코더 (Text Encoder): CLIP
모델: openai/clip-vit-base-patch32

역할: 자연어 질의(visual_instruction)의 의미를 벡터로 임베딩합니다.

선정 이유: 기존 BiGRU보다 복잡한 문장의 맥락을 훨씬 잘 이해하며, 대회 규정상 허용된 가장 강력한 VLM 백본입니다.

2) 이미지 인코더 (Image Encoder): ResNet50
모델: ResNet50 (ImageNet Pre-trained)

역할: 문서 이미지에서 시각적 특징 맵(Feature Map)을 추출합니다.

선정 이유: 베이스라인(ResNet18)보다 더 깊은 층을 사용하여 미세한 시각적 패턴을 포착하며, FPN(Feature Pyramid Network) 스타일의 특징 추출을 수행합니다.

3) 융합 및 예측 (Fusion): Cross-Attention
텍스트 특징(Query)과 이미지 특징(Key/Value)을 Cross-Attention 메커니즘으로 융합하여, 질의와 의미적으로 가장 연관된 이미지 영역의 BBox (cx, cy, w, h)를 예측합니다.

4) 손실 함수 (Loss Function): mIoU 최적화
L1 Loss: 좌표 값의 절대적인 차이를 줄입니다.

GIoU Loss (Generalized IoU): BBox 간의 겹침 정도(IoU)를 직접적으로 최적화하여, 리더보드 평가 지표인 mIoU 점수를 극대화합니다.

2. 실험 환경 (Environment)
본 모델의 학습 및 추론은 다음 환경에서 수행되었습니다.

플랫폼: 엘리스(Elice) 클라우드 GPU

인스턴스 유형: G-NAHP-80

하드웨어:

GPU: NVIDIA A100X (80GB vRAM)

CPU: Intel Xeon

소프트웨어:

Python 3.10

PyTorch 2.x (CUDA 12.2)

Transformers, Torchvision

3. 사전학습 모델 출처 (Pre-trained Models)

CLIP: Hugging Face - openai/clip-vit-base-patch32

ResNet50: PyTorch Torchvision - ResNet50

4. 파일 구성 (File Structure)
📦 code
 ┣ 📜 code444.py    # [학습] 모델 정의, 데이터 로더, 학습 루프 통합 스크립트
 ┣ 📜 test.py       # [추론] 학습된 모델로 제출 파일 생성
 ┣ 📜 requirements.txt # 필요 라이브러리 목록
 ┗ 📜 README.md     # 설명 문서
5. 실행 방법 (Usage)
1) 필수 라이브러리 설치
Bash

pip install -r requirements.txt
2) 모델 학습 (Train)
code444.py를 실행하여 모델을 학습합니다. (데이터셋 1/160 샘플링 또는 전체 학습 설정 가능)

Bash

# Elice 터미널에서 실행
python code444.py
Input: /home/elicer/data/train_valid/train (학습 데이터)

Output: ./outputs/ckpt/best_miou_model.pth (학습된 모델 가중치)

3) 모델 추론 (Inference)
학습된 모델을 사용하여 Test 데이터셋에 대한 예측을 수행하고 제출 파일(test_pred.csv)을 생성합니다.

Bash

# Elice 터미널에서 실행
python test.py \
    --json_dir /home/elicer/open/test/query \
    --jpg_dir /home/elicer/open/test/images \
    --ckpt ./outputs/ckpt/best_miou_model.pth \
    --out_csv ./outputs/preds/test_pred.csv
Input: /home/elicer/open/test (테스트 데이터)

Output: ./outputs/preds/test_pred.csv (최종 제출 파일)
