# 📊 PNU 데이터 분석 실습 (Data Analysis Coursework)

부산대학교 데이터 분석 실습 및 과제 저장소입니다.

> 🌐 **[에디토리얼 웹 발표 슬라이드 덱 바로가기 (Live Slide Deck)](https://m1nseokshin.github.io/PNU-Data/)**  
> 타이타닉 재난 데이터의 구조적 불평등 분석 및 머신러닝 예측 모델 인사이트 프레젠테이션.

---

## 📂 디렉토리 구조

```
├── week_03/                     # 타이타닉 생존자 데이터 분석 및 머신러닝 예측
│   ├── data/                    # 타이타닉 데이터셋 (CSV)
│   ├── download_titanic.py      # KaggleHub 데이터 자동 다운로드 스크립트
│   ├── basic_eda.py             # 결측치, 중복값, 이상치(IQR) 무결성 검증 스크립트
│   ├── predict_survival.py      # EDA 시각화 및 머신러닝 예측 모델 파이프라인
│   ├── titanic_eda_tutorial.ipynb   # 주피터 노트북 1: 데이터 무결성 검증 튜토리얼
│   ├── titanic_ml_pipeline.ipynb    # 주피터 노트북 2: EDA 및 머신러닝 전체 파이프라인
│   ├── eda_result.png           # 결측치/이상치 종합 검증 리포트 이미지
│   ├── eda_survival_analysis.png# 특성별 생존율 분석 시각화 차트
│   └── model_evaluation.png     # 모델 성능 및 피처 중요도 차트
└── week_03_02/                  # 피마 인디언 당뇨병(Pima Indians Diabetes) 데이터 분석
    ├── pima-indians-diabetes.csv    # 당뇨병 데이터셋 (768행 9열)
    ├── data_analysis_basics.py      # 결측치, 중복값, 이상치 검증 스크립트
    ├── data_analysis_basics.ipynb   # 주피터 노트북 실습 (사전 렌더링 완료)
    ├── 01_missing_values_check.png  # 숨은 결측치(0값) 분석 차트
    ├── 02_outliers_boxplot.png      # IQR 기준 이상치 박스플롯
    └── 03_data_quality_summary.png  # 데이터 품질 종합 요약표
```

---

## 🚢 Week 03: 타이타닉 생존자 데이터 분석 & 머신러닝 예측

### 1. 데이터 품질 검증 (Data Integrity)
- **결측치(Missing Values):** `Cabin`(77.1%), `Age`(19.9%), `Embarked`(0.2%) 확인 및 정밀 대체(호칭별 중앙값)
- **중복값(Duplicates):** 0건 확인
- **이상치(Outliers):** 사분위수 범위(IQR) 방식을 통한 `Fare`(운임 요금) 극단값 식별 및 로그 변환

### 2. 탐색적 데이터 분석 (EDA) 주요 통찰
- **성별(Sex):** 여성 생존율(74.2%) vs 남성 생존율(18.9%) ("여성과 아이 먼저" 구조 원칙)
- **객실 등급(Pclass):** 1등석(63.0%) > 2등석(47.3%) > 3등석(24.2%)
- **가족 수(FamilySize):** 2~4인의 소가족 동승객의 생존율(55~70%)이 가장 높음

### 3. 머신러닝 모델 성능 비교

| 모델명 | 테스트 정확도 (Accuracy) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **로지스틱 회귀 (Logistic Regression)** | 82.12% | 75.76% | 0.8663 |
| **의사결정나무 (Decision Tree)** | 80.45% | 71.07% | 0.8465 |
| 🏆 **랜덤 포레스트 (Random Forest)** | **83.24%** | **76.92%** | **0.8616** |

- **가장 결정적인 피처 (Feature Importance):**  
  1위: `Sex_male`(성별), 2위: `Title_Mr`(성인 남성 호칭), 3위: `LogFare`(운임 요금), 4위: `Age`(나이)

---

## 🩺 Week 03-02: 피마 인디언 당뇨병 데이터 무결성 검증

### 1. 결측치(Missing Values)의 함정
- **표면상 NaN 결측치:** 0개
- **도메인 지식 기반 "숨은 결측치":**
  - 생리학적으로 0이 될 수 없는 수치(`Insulin` 48.7%, `SkinThickness` 29.6%, `BloodPressure` 4.6%, `BMI` 1.4%, `Glucose` 0.7%)가 0으로 대체 입력되어 있음을 확인
  - 결측치 위치 히트맵 및 비율 차트 시각화 완료

### 2. 중복값(Duplicates)
- 총 768행 중 중복 수집된 행 0건 확인

### 3. 이상치(Outliers) 검증 (IQR 방식)
- 각 특성별 박스플롯(Boxplot) 시각화
- `Insulin`(34건), `BloodPressure`(45건), `BMI`(19건), `Age`(9건) 등에서 사분위수 정상 범위를 벗어난 이상치 식별

---

## 🛠 실행 방법

```bash
# 가상환경 또는 패키지 설치
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub notebook ipykernel

# Week 03 검증 및 모델 실행
python3 week_03/basic_eda.py
python3 week_03/predict_survival.py

# 주피터 노트북 실행
jupyter notebook
```
