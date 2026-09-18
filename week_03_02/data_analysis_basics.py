import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 한글 폰트 및 스타일 설정 (Mac 환경: AppleGothic)
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="AppleGothic")

# 2. 데이터 불러오기
columns = [
    "Pregnancies",               # 임신 횟수
    "Glucose",                   # 포도당 수치
    "BloodPressure",             # 혈압
    "SkinThickness",             # 삼두근 피부 두께
    "Insulin",                   # 인슐린
    "BMI",                       # 체질량 지수
    "DiabetesPedigreeFunction",  # 당뇨 내력 가중치
    "Age",                       # 나이
    "Outcome"                    # 당뇨 여부 (0: 정상, 1: 당뇨)
]

df = pd.read_csv("pima-indians-diabetes.csv", names=columns)

print("="*60)
print("📌 [1] 데이터 기본 구조 확인")
print("="*60)
print(f"- 데이터 크기 (행, 열): {df.shape}")
print("\n[상위 5개 데이터]")
print(df.head())

print("\n" + "="*60)
print("📌 [2] 중복값(Duplicates) 검증")
print("="*60)
duplicate_count = df.duplicated().sum()
print(f"- 전체 중복 행 개수: {duplicate_count}개")
if duplicate_count == 0:
    print("  -> 중복된 데이터가 없습니다. (정상)")
else:
    print(f"  -> {duplicate_count}개의 중복 행이 발견되었습니다.")

print("\n" + "="*60)
print("📌 [3] 결측치(Missing Values) 검증")
print("="*60)
print("1) 단순 NaN(Null) 결측치 개수:")
print(df.isnull().sum())
print("  -> 표면상으로는 NaN 결측치가 0개입니다.")

print("\n2) 도메인 지식 기반 '숨겨진 결측치(0값)' 검증:")
# 생리학적으로 0이 될 수 없는 컬럼들
zero_features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

zero_counts = {}
for col in zero_features:
    cnt = (df[col] == 0).sum()
    pct = (cnt / len(df)) * 100
    zero_counts[col] = (cnt, pct)
    print(f"  - {col:15s}: {cnt:3d}개 ({pct:5.2f}%) 가 0으로 입력됨")

print("\n" + "="*60)
print("📌 [4] 이상치(Outliers) 검증 (IQR 방식)")
print("="*60)
# IQR = Q3 (75%) - Q1 (25%)
# 정상 범위: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
outlier_summary = {}
feature_cols = columns[:-1] # Outcome 제외

for col in feature_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    
    outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
    outlier_count = len(outliers)
    outlier_summary[col] = (outlier_count, lower_limit, upper_limit)
    print(f"  - {col:25s}: 이상치 {outlier_count:2d}개 | 정상범위 [{lower_limit:6.2f} ~ {upper_limit:6.2f}]")

# ==========================================================
# 📊 시각화 생성
# ==========================================================

# 1) 결측치 및 0값 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (1-1) 0값 비율 바차트
zero_pct_series = pd.Series({k: v[1] for k, v in zero_counts.items()})
colors = sns.color_palette("Reds_r", n_colors=len(zero_pct_series))
bars = axes[0].bar(zero_pct_series.index, zero_pct_series.values, color=colors, edgecolor='black')
axes[0].set_title("생리학적 불가능 수치 (0값 = 숨은 결측치) 비율 (%)", fontsize=13, pad=12, fontweight='bold')
axes[0].set_ylabel("비율 (%)", fontsize=11)
axes[0].set_ylim(0, 60)
for bar in bars:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

# (1-2) 결측치 히트맵 (0을 NaN으로 치환했을 때의 분포)
df_nan = df.copy()
df_nan[zero_features] = df_nan[zero_features].replace(0, np.nan)
sns.heatmap(df_nan[zero_features].isnull(), cbar=False, cmap='YlOrRd', ax=axes[1], yticklabels=False)
axes[1].set_title("숨은 결측치(0) 위치 히트맵 (노란색: 정상, 붉은색: 결측)", fontsize=13, pad=12, fontweight='bold')

plt.tight_layout()
plt.savefig("01_missing_values_check.png", dpi=200)
plt.close()
print("\n[시각화 완료] '01_missing_values_check.png' 저장 완료")

# 2) 이상치 시각화 (Boxplot)
plt.figure(figsize=(15, 10))
for i, col in enumerate(feature_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col], color="skyblue", flierprops=dict(marker='o', markersize=4, markerfacecolor='red', alpha=0.6))
    plt.title(f"{col} 박스플롯", fontsize=12, fontweight='bold')
    plt.ylabel("값")

plt.suptitle("피처별 이상치 검증 (붉은색 점: IQR 기준 이상치)", fontsize=16, y=0.99)
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig("02_outliers_boxplot.png", dpi=200)
plt.close()
print("[시각화 완료] '02_outliers_boxplot.png' 저장 완료")

# 3) 중복값 및 데이터 검증 요약 시각화
summary_df = pd.DataFrame({
    '항목': ['총 행 수', '중복 행 수', '표면상 NaN', '인슐린(0값 결측)', '피하지방(0값 결측)', '이상치(인슐린)', '이상치(혈압)'],
    '개수': [len(df), duplicate_count, df.isnull().sum().sum(), (df['Insulin'] == 0).sum(), (df['SkinThickness'] == 0).sum(), outlier_summary['Insulin'][0], outlier_summary['BloodPressure'][0]],
})

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4A90E2')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#F8F9FA' if row % 2 == 0 else '#FFFFFF')

plt.title("데이터 품질 검증 요약표 (결측치 / 중복값 / 이상치)", fontsize=14, pad=10, fontweight='bold')
plt.tight_layout()
plt.savefig("03_data_quality_summary.png", dpi=200)
plt.close()
print("[시각화 완료] '03_data_quality_summary.png' 저장 완료")
