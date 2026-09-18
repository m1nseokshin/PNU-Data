# 📊 PNU 데이터 분석 실습 (Data Analysis Coursework)

부산대학교 데이터 분석 실습 및 과제 저장소입니다.

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
└── week_03_02/                  # 당뇨병(Pima Indians Diabetes) 데이터 분석
    ├── pima-indians-diabetes.csv
    ├── data_analysis_basics.py
    └── data_analysis_basics.ipynb
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
