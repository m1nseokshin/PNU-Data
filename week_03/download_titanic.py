import os
import shutil
import kagglehub
import pandas as pd

def main():
    print("Titanic 데이터셋 다운로드를 시작합니다 (heptapod/titanic)...")
    
    # 1. Kaggle 공개 데이터셋 다운로드 (별도 대회 동의/인증 불필요)
    try:
        path = kagglehub.dataset_download("heptapod/titanic")
        print(f"Kaggle 캐시 다운로드 경로: {path}")
    except Exception as e:
        print(f"\n[오류 발생] 다운로드 중 문제가 발생했습니다:\n{e}")
        return

    # 2. 다운로드된 파일 확인
    downloaded_files = os.listdir(path)
    print(f"다운로드된 파일 목록: {downloaded_files}")

    # 3. 현재 작업 디렉토리(./data)로 복사
    target_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(target_dir, exist_ok=True)
    for file_name in downloaded_files:
        src = os.path.join(path, file_name)
        dst = os.path.join(target_dir, file_name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    print(f"현재 프로젝트 내 저장 위치: {target_dir}")

    # 4. Pandas로 데이터 미리보기
    for file_name in downloaded_files:
        if file_name.endswith(".csv"):
            csv_path = os.path.join(target_dir, file_name)
            df = pd.read_csv(csv_path)
            print(f"\n[{file_name} 상위 5개 행]")
            print(df.head())
            print(f"\n총 데이터 크기: {df.shape}")

if __name__ == "__main__":
    main()
