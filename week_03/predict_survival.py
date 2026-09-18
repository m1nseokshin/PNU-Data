import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------
# 0. 한글 폰트 설정 (macOS: AppleGothic)
# ----------------------------------------------------
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def run_eda_visualizations(df, output_path="eda_survival_analysis.png"):
    """생존율과 주요 변수(성별, 객실등급, 나이, 가족수 등)의 상관관계 시각화"""
    print("\n" + "="*50)
    print(" 1. 생존 요인 탐색적 데이터 분석(EDA) 시각화 생성 중...")
    print("="*50)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("타이타닉 주요 특성별 생존율 분석 (EDA)", fontsize=18, fontweight='bold', y=0.98)

    # 1) 성별 생존율
    sns.barplot(x='Sex', y='Survived', data=df, ax=axes[0, 0], palette='pastel')
    axes[0, 0].set_title("1. 성별 생존율 (여성 vs 남성)", fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel("생존율 (Survival Rate)")
    axes[0, 0].set_ylim(0, 1.0)
    for p in axes[0, 0].patches:
        h = p.get_height()
        axes[0, 0].annotate(f'{h*100:.1f}%', (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3), textcoords='offset points')

    # 2) 객실 등급별 생존율
    sns.barplot(x='Pclass', y='Survived', data=df, ax=axes[0, 1], palette='Blues_d')
    axes[0, 1].set_title("2. 객실 등급(Pclass)별 생존율", fontsize=13, fontweight='bold')
    axes[0, 1].set_ylabel("생존율")
    axes[0, 1].set_ylim(0, 1.0)
    for p in axes[0, 1].patches:
        h = p.get_height()
        axes[0, 1].annotate(f'{h*100:.1f}%', (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3), textcoords='offset points')

    # 3) 성별 x 객실 등급 복합 생존율
    sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df, ax=axes[0, 2], palette='Set2')
    axes[0, 2].set_title("3. 객실 등급 × 성별 복합 생존율", fontsize=13, fontweight='bold')
    axes[0, 2].set_ylabel("생존율")
    axes[0, 2].set_ylim(0, 1.0)

    # 4) 연령대별 생존/사망 분포 (KDE)
    sns.kdeplot(df[df['Survived'] == 1]['Age'].dropna(), ax=axes[1, 0], label='생존 (Survived)', fill=True, color='#2ecc71')
    sns.kdeplot(df[df['Survived'] == 0]['Age'].dropna(), ax=axes[1, 0], label='사망 (Deceased)', fill=True, color='#e74c3c')
    axes[1, 0].set_title("4. 나이대별 생존/사망자 분포", fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel("나이 (Age)")
    axes[1, 0].legend()

    # 5) 가족 수(FamilySize = SibSp + Parch + 1)별 생존율
    df_temp = df.copy()
    df_temp['FamilySize'] = df_temp['SibSp'] + df_temp['Parch'] + 1
    sns.barplot(x='FamilySize', y='Survived', data=df_temp, ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title("5. 동승 가족 수(FamilySize)별 생존율", fontsize=13, fontweight='bold')
    axes[1, 1].set_ylabel("생존율")
    axes[1, 1].set_ylim(0, 1.0)

    # 6) 탑승 항구(Embarked)별 생존율
    sns.barplot(x='Embarked', y='Survived', data=df, ax=axes[1, 2], palette='mako')
    axes[1, 2].set_title("6. 탑승 항구(Embarked)별 생존율 (C/Q/S)", fontsize=13, fontweight='bold')
    axes[1, 2].set_ylabel("생존율")
    axes[1, 2].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"EDA 시각화 저장 완료: {output_path}")


def preprocess_features(df):
    """결측치 처리 및 머신러닝 피처 엔지니어링"""
    print("\n" + "="*50)
    print(" 2. 데이터 전처리 및 피처 엔지니어링 수행 중...")
    print("="*50)

    data = df.copy()

    # 1) 이름에서 호칭(Title) 추출 (Mr, Miss, Mrs, Master 등)
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # 희귀 호칭 묶기
    common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    data['Title'] = data['Title'].apply(lambda x: x if x in common_titles else 'Rare')

    # 2) 나이(Age) 결측치: 호칭(Title)별 중앙값으로 정밀 대체
    age_medians = data.groupby('Title')['Age'].median()
    for title, median_age in age_medians.items():
        data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = median_age
    print(f"- 나이(Age) 결측치 대체 완료 (호칭별 중앙값: {dict(age_medians.round(1))})")

    # 3) 탑승항구(Embarked) 결측치: 최빈값 'S'로 대체
    mode_embarked = data['Embarked'].mode()[0]
    data['Embarked'] = data['Embarked'].fillna(mode_embarked)
    print(f"- 탑승항구(Embarked) 결측치 대체 완료 (최빈값: '{mode_embarked}')")

    # 4) 운임요금(Fare) 결측치 및 스케일링: 결측치는 중앙값, 로그 변환(np.log1p)
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['LogFare'] = np.log1p(data['Fare'])

    # 5) 가족 수(FamilySize) 및 1인 승객 여부(IsAlone) 생성
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # 6) 범주형 데이터 원-핫 인코딩 (One-Hot Encoding)
    # Sex: female/male -> 1/0
    data['Sex_male'] = (data['Sex'] == 'male').astype(int)

    # Embarked, Title 원-핫 인코딩
    data = pd.get_dummies(data, columns=['Embarked', 'Title'], drop_first=True, dtype=int)

    # 모델 학습에 사용할 피처 선정
    feature_cols = [
        'Pclass', 'Sex_male', 'Age', 'LogFare', 'FamilySize', 'IsAlone',
        'Embarked_Q', 'Embarked_S',
        'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare'
    ]
    # 존재하는 컬럼만 선택
    feature_cols = [c for c in feature_cols if c in data.columns]

    X = data[feature_cols]
    y = data['Survived']

    print(f"- 최종 입력 특성({len(feature_cols)}개): {feature_cols}")
    print(f"- 데이터셋 크기: X={X.shape}, y={y.shape}")

    return X, y, feature_cols


def train_and_evaluate_models(X, y, output_path="model_evaluation.png"):
    """머신러닝 모델 학습, 성능 평가 및 시각화"""
    print("\n" + "="*50)
    print(" 3. 머신러닝 예측 모델 학습 및 평가")
    print("="*50)

    # Train / Test 데이터셋 분리 (80% 학습, 20% 테스트)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"- 학습 데이터(Train): {X_train.shape[0]}건 | 테스트 데이터(Test): {X_test.shape[0]}건")

    # 비교할 3가지 분류 모델
    models = {
        "로지스틱 회귀 (Logistic Regression)": LogisticRegression(max_iter=1000, random_state=42),
        "의사결정나무 (Decision Tree)": DecisionTreeClassifier(max_depth=5, random_state=42),
        "랜덤 포레스트 (Random Forest)": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }

    results = []
    trained_models = {}
    y_preds = {}
    y_probs = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        trained_models[name] = model
        y_preds[name] = y_pred
        y_probs[name] = y_prob

        results.append({
            "모델명": name,
            "정확도(Accuracy)": f"{acc*100:.2f}%",
            "정밀도(Precision)": f"{prec*100:.2f}%",
            "재현율(Recall)": f"{rec*100:.2f}%",
            "F1-Score": f"{f1*100:.2f}%",
            "ROC-AUC": f"{auc:.4f}",
            "acc_raw": acc
        })

    perf_df = pd.DataFrame(results)
    print("\n[모델별 테스트 성능 비교]")
    print(perf_df[["모델명", "정확도(Accuracy)", "정밀도(Precision)", "재현율(Recall)", "F1-Score", "ROC-AUC"]].to_string(index=False))

    # 최고 성능 모델 선정 (랜덤 포레스트)
    best_model_name = "랜덤 포레스트 (Random Forest)"
    best_model = trained_models[best_model_name]
    best_y_pred = y_preds[best_model_name]

    print(f"\n[{best_model_name} 상세 분류 리포트]")
    print(classification_report(y_test, best_y_pred, target_names=["사망(0)", "생존(1)"]))

    # ----------------------------------------------------
    # 시각화: 모델 평가 종합 대시보드 (4개 서브플롯)
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("타이타닉 생존 예측 모델 성능 및 특성 중요도 분석", fontsize=18, fontweight='bold', y=0.98)

    # 1) 모델별 정확도 비교 막대 그래프
    acc_series = [r['acc_raw'] * 100 for r in results]
    names_short = ["Logistic Reg.", "Decision Tree", "Random Forest"]
    bars = axes[0, 0].bar(names_short, acc_series, color=['#3498db', '#e67e22', '#2ecc71'])
    axes[0, 0].set_title("1. 모델별 테스트 정확도(Accuracy) 비교", fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel("정확도 (%)")
    axes[0, 0].set_ylim(70, 95)
    for bar in bars:
        h = bar.get_height()
        axes[0, 0].annotate(f'{h:.2f}%', (bar.get_x() + bar.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3), textcoords='offset points')

    # 2) Confusion Matrix (최고 모델: Random Forest)
    cm = confusion_matrix(y_test, best_y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False, annot_kws={"size": 14, "fontweight": "bold"})
    axes[0, 1].set_title(f"2. 혼동 행렬 (Confusion Matrix: {best_model_name})", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel("예측값 (Predicted)")
    axes[0, 1].set_ylabel("실제값 (Actual)")
    axes[0, 1].set_xticklabels(["사망 (0)", "생존 (1)"])
    axes[0, 1].set_yticklabels(["사망 (0)", "생존 (1)"])

    # 3) ROC Curve 비교
    for name in models.keys():
        fpr, tpr, _ = roc_curve(y_test, y_probs[name])
        auc = roc_auc_score(y_test, y_probs[name])
        axes[1, 0].plot(fpr, tpr, lw=2, label=f"{name.split(' ')[0]} (AUC = {auc:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')
    axes[1, 0].set_title("3. ROC 곡선 (ROC-AUC Curve)", fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel("위양성률 (False Positive Rate)")
    axes[1, 0].set_ylabel("진양성률 (True Positive Rate)")
    axes[1, 0].legend(loc='lower right')

    # 4) 랜덤 포레스트 피처 중요도 (Feature Importance)
    feature_importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True)
    feature_importances.plot(kind='barh', ax=axes[1, 1], color='#8e44ad')
    axes[1, 1].set_title("4. 특성 중요도 (Feature Importance: 어떤 요인이 생존을 좌우했나?)", fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel("상대적 중요도")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"모델 평가 시각화 저장 완료: {output_path}")

    return best_model, X.columns


def sample_prediction(model, feature_names):
    """가상의 승객 데이터로 생존 확률 예측 테스트"""
    print("\n" + "="*50)
    print(" 4. 가상 승객 생존 확률 예측 시뮬레이션")
    print("="*50)

    # 가상 승객 1: 1등석, 25세 여성, 고액 운임, 혼자 탑승 (Miss)
    # 가상 승객 2: 3등석, 28세 남성, 저렴한 운임, 혼자 탑승 (Mr)
    sample_passengers = pd.DataFrame([
        {
            'Pclass': 1, 'Sex_male': 0, 'Age': 25.0, 'LogFare': np.log1p(100.0),
            'FamilySize': 1, 'IsAlone': 1, 'Embarked_Q': 0, 'Embarked_S': 1,
            'Title_Miss': 1, 'Title_Mr': 0, 'Title_Mrs': 0, 'Title_Rare': 0
        },
        {
            'Pclass': 3, 'Sex_male': 1, 'Age': 28.0, 'LogFare': np.log1p(8.0),
            'FamilySize': 1, 'IsAlone': 1, 'Embarked_Q': 0, 'Embarked_S': 1,
            'Title_Miss': 0, 'Title_Mr': 1, 'Title_Mrs': 0, 'Title_Rare': 0
        }
    ])

    for col in feature_names:
        if col not in sample_passengers.columns:
            sample_passengers[col] = 0
    sample_passengers = sample_passengers[feature_names]

    probs = model.predict_proba(sample_passengers)

    print("👩 승객 1 (1등석 25세 여성 Miss):")
    print(f"   -> 생존 확률: {probs[0][1]*100:.1f}% (예측 결과: {'생존' if probs[0][1] > 0.5 else '사망'})")

    print("\n👨 승객 2 (3등석 28세 남성 Mr):")
    print(f"   -> 생존 확률: {probs[1][1]*100:.1f}% (예측 결과: {'생존' if probs[1][1] > 0.5 else '사망'})")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "Titanic-Dataset.csv")
    df = pd.read_csv(data_path)

    # 1. EDA 시각화
    eda_png = os.path.join(base_dir, "eda_survival_analysis.png")
    run_eda_visualizations(df, eda_png)

    # 2. 전처리 및 피처 엔지니어링
    X, y, feature_names = preprocess_features(df)

    # 3. 모델 학습 및 평가 시각화
    model_png = os.path.join(base_dir, "model_evaluation.png")
    best_model, feature_names = train_and_evaluate_models(X, y, model_png)

    # 4. 가상 승객 예측 시뮬레이션
    sample_prediction(best_model, feature_names)


if __name__ == "__main__":
    main()
