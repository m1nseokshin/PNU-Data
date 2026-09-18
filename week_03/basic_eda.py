import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 0. 한글 폰트 설정 (macOS 환경: AppleGothic)
# ==========================================
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def load_dataset():
    """데이터셋을 로드합니다 (우선 원본 Titanic-Dataset.csv 시도)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "data", "Titanic-Dataset.csv")
    alt_path = os.path.join(base_dir, "data", "train_and_test2.csv")

    if os.path.exists(raw_path):
        print(f"[1] 데이터 로드 완료: {raw_path}")
        return pd.read_csv(raw_path)
    elif os.path.exists(alt_path):
        print(f"[1] 데이터 로드 완료: {alt_path}")
        return pd.read_csv(alt_path)
    else:
        raise FileNotFoundError("데이터셋을 찾을 수 없습니다. ./data 폴더를 확인해주세요.")


def inspect_basic_info(df):
    """기본 데이터 크기 및 구조 확인"""
    print("\n" + "="*50)
    print(" 1. 기본 데이터 구조 파악 (Shape & Info)")
    print("="*50)
    print(f"- 전체 데이터 크기: {df.shape[0]}행 × {df.shape[1]}열")
    print("\n[컬럼별 데이터 타입 및 결측치 현황]")
    print(df.dtypes)


def check_missing_values(df):
    """결측치(Missing Values / NaN) 검증"""
    print("\n" + "="*50)
    print(" 2. 결측치(Missing Values) 검증")
    print("="*50)
    missing_count = df.isnull().sum()
    missing_ratio = (missing_count / len(df)) * 100

    missing_df = pd.DataFrame({
        "결측치 개수": missing_count,
        "결측치 비율(%)": missing_ratio.round(2)
    })
    # 결측치가 있는 컬럼만 필터링
    missing_only = missing_df[missing_df["결측치 개수"] > 0].sort_values(by="결측치 개수", ascending=False)
    
    if len(missing_only) > 0:
        print(missing_only)
    else:
        print("결측치가 존재하는 컬럼이 없습니다.")
    return missing_df


def check_duplicates(df):
    """중복값(Duplicates) 검증"""
    print("\n" + "="*50)
    print(" 3. 중복값(Duplicate Rows) 검증")
    print("="*50)
    total_rows = len(df)
    duplicate_count = df.duplicated().sum()
    print(f"- 전체 행 수: {total_rows}")
    print(f"- 완전히 동일한 중복 행 수: {duplicate_count}건 ({(duplicate_count/total_rows)*100:.2f}%)")

    # 식별자 컬럼(PassengerId 등)이 있다면 식별자 중복도 검사
    id_cols = [c for c in df.columns if 'id' in c.lower()]
    for id_col in id_cols:
        id_dups = df[id_col].duplicated().sum()
        print(f"- 고유 식별자 [{id_col}] 중복: {id_dups}건")
    return duplicate_count


def check_outliers_iqr(df, numeric_cols):
    """IQR(사분위 범위) 방식으로 이상치(Outliers) 탐지"""
    print("\n" + "="*50)
    print(" 4. 이상치(Outliers) 검증 (IQR 방식)")
    print("="*50)
    outlier_summary = []

    for col in numeric_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = len(outliers)
        outlier_ratio = (outlier_count / len(series)) * 100

        outlier_summary.append({
            "컬럼명": col,
            "Q1(25%)": round(q1, 2),
            "Q3(75%)": round(q3, 2),
            "IQR": round(iqr, 2),
            "정상 하한값": round(lower_bound, 2),
            "정상 상한값": round(upper_bound, 2),
            "이상치 개수": outlier_count,
            "이상치 비율(%)": round(outlier_ratio, 2)
        })

    summary_df = pd.DataFrame(outlier_summary)
    print(summary_df.to_string(index=False))
    return summary_df


def visualize_all(df, missing_df, duplicate_count, save_path="eda_result.png"):
    """결측치, 중복값, 이상치를 6개 서브플롯으로 시각화"""
    print("\n" + "="*50)
    print(f" 5. 시각화 생성 중... -> {save_path}")
    print("="*50)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("데이터 무결성 검증 리포트 (결측치 · 중복값 · 이상치)", fontsize=18, fontweight='bold', y=0.98)

    # ----------------------------------------------------
    # 1) 결측치 히트맵 (행별 결측치 분포 시각화)
    # ----------------------------------------------------
    ax1 = fig.add_subplot(2, 3, 1)
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='magma', ax=ax1)
    ax1.set_title("1. 결측치 분포 히트맵 (밝은 선 = 결측치)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("컬럼명")
    ax1.set_ylabel("데이터 행 (인덱스)")

    # ----------------------------------------------------
    # 2) 컬럼별 결측치 비율 막대 그래프
    # ----------------------------------------------------
    ax2 = fig.add_subplot(2, 3, 2)
    missing_filtered = missing_df[missing_df["결측치 비율(%)"] > 0].sort_values(by="결측치 비율(%)", ascending=False)
    if len(missing_filtered) > 0:
        bars = ax2.bar(missing_filtered.index, missing_filtered["결측치 비율(%)"], color=['#e74c3c', '#e67e22', '#f1c40f'][:len(missing_filtered)])
        ax2.set_title("2. 컬럼별 결측치 비율 (%)", fontsize=13, fontweight='bold')
        ax2.set_ylabel("결측 비율 (%)")
        ax2.set_ylim(0, 100)
        # 막대 위에 수치 텍스트 표시
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "결측치 없음 (100% 완전함)", ha='center', va='center', fontsize=14, color='green')
        ax2.set_title("2. 컬럼별 결측치 비율 (%)", fontsize=13, fontweight='bold')

    # ----------------------------------------------------
    # 3) 중복값 비율 파이 차트
    # ----------------------------------------------------
    ax3 = fig.add_subplot(2, 3, 3)
    total_count = len(df)
    unique_count = total_count - duplicate_count
    ax3.pie(
        [unique_count, duplicate_count],
        labels=["고유 행(Unique)", f"중복 행(Duplicate: {duplicate_count}건)"],
        autopct='%1.1f%%',
        startangle=140,
        colors=['#2ecc71', '#e74c3c'],
        explode=(0, 0.15) if duplicate_count > 0 else (0, 0)
    )
    ax3.set_title("3. 데이터 행 중복값 검증", fontsize=13, fontweight='bold')

    # ----------------------------------------------------
    # 4) 이상치 - 나이(Age) Boxplot
    # ----------------------------------------------------
    ax4 = fig.add_subplot(2, 3, 4)
    target_age = 'Age' if 'Age' in df.columns else None
    flier_style = dict(marker='o', markerfacecolor='#e74c3c', markeredgecolor='#c0392b', markersize=6, alpha=0.7)
    if target_age:
        sns.boxplot(x=df[target_age], ax=ax4, color='#3498db', flierprops=flier_style)
        ax4.set_title(f"4. 이상치 확인: {target_age} (Boxplot)", fontsize=13, fontweight='bold')
        ax4.set_xlabel(f"{target_age} (빨간 점 = IQR 기준 이상치)")
    else:
        ax4.text(0.5, 0.5, "Age 컬럼 없음", ha='center', va='center')

    # ----------------------------------------------------
    # 5) 이상치 - 운임 요금(Fare) Boxplot
    # ----------------------------------------------------
    ax5 = fig.add_subplot(2, 3, 5)
    target_fare = 'Fare' if 'Fare' in df.columns else None
    if target_fare:
        sns.boxplot(x=df[target_fare], ax=ax5, color='#9b59b6', flierprops=flier_style)
        ax5.set_title(f"5. 이상치 확인: {target_fare} (Boxplot)", fontsize=13, fontweight='bold')
        ax5.set_xlabel(f"{target_fare} (빨간 점 = 고액 요금 이상치)")
    else:
        ax5.text(0.5, 0.5, "Fare 컬럼 없음", ha='center', va='center')

    # ----------------------------------------------------
    # 6) 이상치 분포 확인 - Fare 히스토그램 및 KDE (왜도 시각화)
    # ----------------------------------------------------
    ax6 = fig.add_subplot(2, 3, 6)
    if target_fare:
        sns.histplot(df[target_fare].dropna(), kde=True, ax=ax6, color='#9b59b6', bins=30)
        ax6.set_title(f"6. {target_fare} 분포 (우측 꼬리가 긴 왜도)", fontsize=13, fontweight='bold')
        ax6.set_xlabel("운임 요금 (Fare)")
        ax6.set_ylabel("승객 수")
    else:
        ax6.text(0.5, 0.5, "Fare 컬럼 없음", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"시각화 그래프가 성공적으로 저장되었습니다: {save_path}")


def main():
    df = load_dataset()

    # 1. 기본 정보
    inspect_basic_info(df)

    # 2. 결측치
    missing_df = check_missing_values(df)

    # 3. 중복값
    duplicate_count = check_duplicates(df)

    # 4. 이상치 (수치형 컬럼 선정: Age, Fare 등)
    candidate_cols = [col for col in ['Age', 'Fare', 'SibSp', 'Parch'] if col in df.columns]
    check_outliers_iqr(df, candidate_cols)

    # 5. 시각화 생성 및 저장
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_png = os.path.join(current_dir, "eda_result.png")
    visualize_all(df, missing_df, duplicate_count, save_path=output_png)


if __name__ == "__main__":
    main()
