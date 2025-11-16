import urllib.request
import zipfile
import os

data_url = "다운링크^^"
data_dir = "./data"  # 데이터 디렉터리 경로
zip_path = f"{data_dir}/train_valid.zip"

# 이미 다운로드되어 있는지 확인
if not os.path.exists(zip_path):
    print("📥 데이터 다운로드 중... (시간이 걸릴 수 있습니다)")
    
    # --- (수정된 부분) ---
    # 파일을 저장할 디렉터리를 생성합니다.
    # exist_ok=True는 디렉터리가 이미 있어도 오류를 발생시키지 않습니다.
    os.makedirs(data_dir, exist_ok=True)
    # --------------------
    
    urllib.request.urlretrieve(data_url, zip_path)
    print("✅ 다운로드 완료!")
else:
    print("✅ 데이터가 이미 다운로드되어 있습니다.")

# (이후 코드 ... )
# 압축 해제
if not os.path.exists(f"./data"):
    print("📦 압축 해제 중...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(f"./data")
    print("✅ 압축 해제 완료!")
else:
    print("✅ 데이터가 이미 압축 해제되어 있습니다.")
