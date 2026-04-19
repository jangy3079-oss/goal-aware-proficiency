"""
Baseline ML Model for Language Proficiency Assessment
======================================================
Bamdev et al. (2023) 논문 방식 재현
"Automated Speech Scoring System Under The Lens"
IJAIED, arXiv:2111.15156

논문의 핵심:
  1. 유창성/발음/내용/문법어휘/음향 5개 카테고리 피처 추출
  2. ML 모델(Ridge, SVR, Random Forest)로 총점 예측
  3. SHAP으로 각 피처 기여도 수치화

이 코드에서는:
  - Speechocean762의 sentence-level 점수를 피처로 사용
    (실제 논문은 음성에서 직접 추출하지만, 여기서는 레이블을 피처로 써서
     ML 파이프라인 구조와 평가 방법론을 먼저 익히는 용도)
  - 이후 단계에서 실제 음성 피처(MFCC, GOP 등)로 교체 예정
"""

import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from pathlib import Path
from typing import Dict, Tuple

from datasets import load_dataset
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# ── 재현성 고정 ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드 & 피처 엔지니어링
# ══════════════════════════════════════════════════════════════════════════════

def load_speechocean(verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    HuggingFace에서 speechocean762 로드 후 DataFrame 변환.

    Returns:
        train_df, test_df : 피처 + 타겟이 포함된 DataFrame
    """
    if verbose:
        print("[1/4] Loading speechocean762 from HuggingFace...")

    dataset = load_dataset("mispeech/speechocean762")

    def to_df(split) -> pd.DataFrame:
        # 오디오 컬럼은 지금 단계에서 불필요 → 제거
        sp = split.remove_columns(['audio']) if 'audio' in split.column_names else split
        records = []
        for item in sp:
            # ── 문장 레벨 점수 (직접 제공됨) ──────────────────────────────
            row = {
                # 타겟
                'total'        : float(item['total']),
                # 피처 그룹 A: 발음 관련
                'accuracy'     : float(item['accuracy']),
                'prosodic'     : float(item['prosodic']),
                # 피처 그룹 B: 유창성 관련
                'fluency'      : float(item['fluency']),
                # 피처 그룹 C: 완성도
                'completeness' : float(item['completeness']),
                # 메타
                'text'         : item['text'],
                'word_count'   : len(item['words']),
            }

            # ── 단어 레벨 집계 피처 (word-level → sentence-level 집계) ──────
            word_accuracies  = [w['accuracy'] for w in item['words']]
            word_totals      = [w['total']    for w in item['words']]
            phone_accuracies = []
            for w in item['words']:
                phone_accuracies.extend(w['phones-accuracy'])

            row['word_acc_mean']   = np.mean(word_accuracies)   # 단어 정확도 평균
            row['word_acc_std']    = np.std(word_accuracies)    # 단어 정확도 분산
            row['word_acc_min']    = np.min(word_accuracies)    # 가장 틀린 단어
            row['word_total_mean'] = np.mean(word_totals)
            row['phone_acc_mean']  = np.mean(phone_accuracies)  # 음소 정확도 평균
            row['phone_acc_std']   = np.std(phone_accuracies)   # 음소 정확도 분산
            row['phone_acc_min']   = np.min(phone_accuracies)   # 가장 틀린 음소

            records.append(row)

        return pd.DataFrame(records)

    train_df = to_df(dataset['train'])
    test_df  = to_df(dataset['test'])

    if verbose:
        print(f"    Train: {len(train_df)} samples | Test: {len(test_df)} samples")
        print(f"    Features: {len(FEATURE_COLS)} | Target: total score (0-10)")

    return train_df, test_df


# 피처 컬럼 정의 (논문의 피처 카테고리에 맞춰 그룹화)
FEATURE_COLS = [
    # Group A: Pronunciation
    'accuracy', 'prosodic',
    # Group B: Fluency
    'fluency',
    # Group C: Completeness
    'completeness',
    # Group D: Word-level aggregates (새로 추가한 피처)
    'word_acc_mean', 'word_acc_std', 'word_acc_min',
    'word_total_mean',
    # Group E: Phone-level aggregates
    'phone_acc_mean', 'phone_acc_std', 'phone_acc_min',
]

TARGET_COL = 'total'

FEATURE_GROUPS = {
    'Pronunciation' : ['accuracy', 'prosodic'],
    'Fluency'       : ['fluency'],
    'Completeness'  : ['completeness'],
    'Word-level'    : ['word_acc_mean', 'word_acc_std', 'word_acc_min', 'word_total_mean'],
    'Phone-level'   : ['phone_acc_mean', 'phone_acc_std', 'phone_acc_min'],
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. 평가 지표
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    회귀 평가 지표 계산.

    논문에서 쓰는 주요 지표:
      - PCC  : Pearson Correlation Coefficient (예측-실제 선형 상관)
      - MSE  : Mean Squared Error
      - MAE  : Mean Absolute Error
      - R2   : 결정계수
    """
    pcc, _  = scipy_stats.pearsonr(y_true, y_pred)
    mse     = mean_squared_error(y_true, y_pred)
    mae     = mean_absolute_error(y_true, y_pred)
    r2      = r2_score(y_true, y_pred)
    rmse    = np.sqrt(mse)
    return {
        'PCC' : round(float(pcc),  4),
        'MSE' : round(float(mse),  4),
        'RMSE': round(float(rmse), 4),
        'MAE' : round(float(mae),  4),
        'R2'  : round(float(r2),   4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. 모델 정의
# ══════════════════════════════════════════════════════════════════════════════

def build_models() -> Dict:
    """
    Bamdev(2023) 논문에서 사용한 모델들 + 추가 모델.
    StandardScaler → 모델 파이프라인으로 구성.
    """
    return {
        # 선형 모델 (해석 용이, 논문 baseline으로 자주 쓰임)
        'Ridge'           : Pipeline([('scaler', StandardScaler()),
                                       ('model',  Ridge(alpha=1.0, random_state=SEED))]),
        'Lasso'           : Pipeline([('scaler', StandardScaler()),
                                       ('model',  Lasso(alpha=0.1, random_state=SEED))]),
        # SVM (Bamdev 2023의 핵심 모델 중 하나)
        'SVR'             : Pipeline([('scaler', StandardScaler()),
                                       ('model',  SVR(kernel='rbf', C=10, gamma='scale'))]),
        # 트리 기반 앙상블
        'RandomForest'    : Pipeline([('scaler', StandardScaler()),
                                       ('model',  RandomForestRegressor(
                                           n_estimators=100, random_state=SEED, n_jobs=-1))]),
        'GradientBoosting': Pipeline([('scaler', StandardScaler()),
                                       ('model',  GradientBoostingRegressor(
                                           n_estimators=100, random_state=SEED))]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. 학습 & 평가
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    train_df : pd.DataFrame,
    test_df  : pd.DataFrame,
    verbose  : bool = True,
) -> pd.DataFrame:
    """
    모든 모델을 학습하고 train/test 성능을 반환.

    Returns:
        results_df : 모델별 평가 지표 DataFrame
    """
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[TARGET_COL].values

    models  = build_models()
    results = []

    if verbose:
        print("\n[2/4] Training models...")
        print(f"    {'Model':<20} {'PCC':>6} {'RMSE':>7} {'MAE':>7} {'R2':>7}")
        print("    " + "-" * 48)

    for name, pipeline in models.items():
        # 5-Fold 교차검증 (train set 내부)
        kf     = KFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_pcc = cross_val_score(pipeline, X_train, y_train,
                                  cv=kf, scoring='r2')  # sklearn의 r2 사용

        # 최종 학습 & 테스트셋 평가
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        metrics = evaluate(y_test, y_pred)

        results.append({
            'Model'    : name,
            'CV_R2_mean': round(cv_pcc.mean(), 4),
            'CV_R2_std' : round(cv_pcc.std(),  4),
            **{f'Test_{k}': v for k, v in metrics.items()}
        })

        if verbose:
            print(f"    {name:<20} {metrics['PCC']:>6.4f} "
                  f"{metrics['RMSE']:>7.4f} {metrics['MAE']:>7.4f} "
                  f"{metrics['R2']:>7.4f}")

    return pd.DataFrame(results), models, (X_train, y_train, X_test, y_test)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 피처 중요도 분석 (SHAP 대신 내장 중요도 + Permutation)
# ══════════════════════════════════════════════════════════════════════════════

def feature_importance_analysis(
    models      : Dict,
    X_train     : np.ndarray,
    y_train     : np.ndarray,
    verbose     : bool = True,
) -> pd.DataFrame:
    """
    RandomForest의 feature importance와
    Ridge의 계수(coefficient)로 피처 기여도 분석.

    Bamdev(2023)에서 SHAP을 쓴 것과 같은 목적:
    '어떤 피처가 점수에 얼마나 기여하는가'를 수치화.
    """
    if verbose:
        print("\n[3/4] Feature importance analysis...")

    imp_records = []

    # ── RandomForest importance ───────────────────────────────────────────────
    rf = models['RandomForest']
    rf.fit(X_train, y_train)
    rf_imp = rf.named_steps['model'].feature_importances_

    # ── Ridge coefficient (절댓값 = 기여도 크기) ──────────────────────────────
    ridge = models['Ridge']
    ridge.fit(X_train, y_train)
    ridge_coef = np.abs(ridge.named_steps['model'].coef_)
    ridge_coef_norm = ridge_coef / ridge_coef.sum()   # 합이 1이 되도록 정규화

    for i, feat in enumerate(FEATURE_COLS):
        # 어떤 그룹에 속하는지
        group = next((g for g, fs in FEATURE_GROUPS.items() if feat in fs), 'Other')
        imp_records.append({
            'Feature'        : feat,
            'Group'          : group,
            'RF_Importance'  : round(float(rf_imp[i]),         4),
            'Ridge_Coef_Norm': round(float(ridge_coef_norm[i]), 4),
        })

    imp_df = pd.DataFrame(imp_records).sort_values('RF_Importance', ascending=False)

    if verbose:
        print(f"    {'Feature':<22} {'Group':<16} {'RF Imp':>8} {'Ridge Coef':>10}")
        print("    " + "-" * 60)
        for _, row in imp_df.iterrows():
            bar = '█' * int(row['RF_Importance'] * 40)
            print(f"    {row['Feature']:<22} {row['Group']:<16} "
                  f"{row['RF_Importance']:>8.4f} {row['Ridge_Coef_Norm']:>10.4f}  {bar}")

    return imp_df


# ══════════════════════════════════════════════════════════════════════════════
# 6. 목적별 가중치 시뮬레이션
# ══════════════════════════════════════════════════════════════════════════════

# 목적별 가중치 (논문 contribution의 핵심)
# 근거: EDA에서 확인한 피처-총점 상관관계 + 언어 교육 이론
#   - 여행: 발음 정확도(0.9490)와 유창성(0.8513)이 높아야 의사소통 가능
#   - 취업: 어휘 완성도 + 정확도 중시 (격식체, 전문 어휘)
#   - 학업: 운율 + 완성도 중시 (발표, 프레젠테이션)
PURPOSE_WEIGHTS = {
    'Equal (Baseline)': {
        'accuracy': 0.25, 'completeness': 0.25,
        'fluency' : 0.25, 'prosodic'    : 0.25
    },
    'Travel': {
        'accuracy': 0.40, 'completeness': 0.10,
        'fluency' : 0.30, 'prosodic'    : 0.20
    },
    'Business/Job': {
        'accuracy': 0.30, 'completeness': 0.40,
        'fluency' : 0.20, 'prosodic'    : 0.10
    },
    'Academic': {
        'accuracy': 0.20, 'completeness': 0.30,
        'fluency' : 0.20, 'prosodic'    : 0.30
    },
}

SENTENCE_FEATS = ['accuracy', 'completeness', 'fluency', 'prosodic']


def compute_purpose_scores(df: pd.DataFrame) -> pd.DataFrame:
    """목적별 가중 점수 계산."""
    out = df.copy()
    for purpose, w in PURPOSE_WEIGHTS.items():
        col = f'score_{purpose.replace("/","_").replace(" ","_")}'
        out[col] = sum(out[f] * w[f] for f in SENTENCE_FEATS)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 7. 시각화
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(
    results_df  : pd.DataFrame,
    imp_df      : pd.DataFrame,
    test_df     : pd.DataFrame,
    best_model  : str,
    models      : Dict,
    X_test      : np.ndarray,
    y_test      : np.ndarray,
    save_dir    : str = '.',
) -> None:
    """4개 서브플롯으로 구성된 종합 결과 그래프."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Baseline ML Model Results — Speechocean762',
                 fontsize=15, fontweight='bold', y=1.01)

    palette = ['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974']

    # ── (A) 모델 비교: Test PCC ───────────────────────────────────────────────
    ax = axes[0][0]
    models_sorted = results_df.sort_values('Test_PCC', ascending=True)
    colors_bar    = ['#d62728' if m == best_model else '#4C72B0'
                     for m in models_sorted['Model']]
    bars = ax.barh(models_sorted['Model'], models_sorted['Test_PCC'],
                   color=colors_bar, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, models_sorted['Test_PCC']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Pearson Correlation Coefficient (PCC)')
    ax.set_title('(A) Model Comparison — Test PCC', fontweight='bold')
    ax.set_xlim(0, 1.08)
    ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.5, linewidth=1.2,
               label='PCC = 0.90')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # ── (B) 피처 중요도 (RandomForest) ────────────────────────────────────────
    ax = axes[0][1]
    imp_plot = imp_df.head(10).sort_values('RF_Importance', ascending=True)
    group_color_map = {
        'Pronunciation': '#4C72B0',
        'Fluency'      : '#55A868',
        'Completeness' : '#C44E52',
        'Word-level'   : '#8172B2',
        'Phone-level'  : '#CCB974',
    }
    bar_colors = [group_color_map.get(g, '#888') for g in imp_plot['Group']]
    ax.barh(imp_plot['Feature'], imp_plot['RF_Importance'],
            color=bar_colors, alpha=0.85, edgecolor='white')
    ax.set_xlabel('Feature Importance (RandomForest)')
    ax.set_title('(B) Top-10 Feature Importance', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    # 범례
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.85)
               for c in group_color_map.values()]
    ax.legend(handles, group_color_map.keys(), fontsize=8,
              title='Feature Group', title_fontsize=8)

    # ── (C) Best model: Predicted vs Actual ───────────────────────────────────
    ax = axes[1][0]
    y_pred_best = models[best_model].predict(X_test)
    ax.scatter(y_test, y_pred_best, alpha=0.3, s=12, color='#4C72B0')
    lims = [min(y_test.min(), y_pred_best.min()) - 0.5,
            max(y_test.max(), y_pred_best.max()) + 0.5]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    pcc_best = results_df.loc[results_df['Model']==best_model, 'Test_PCC'].values[0]
    rmse_best = results_df.loc[results_df['Model']==best_model, 'Test_RMSE'].values[0]
    ax.set_xlabel('Actual Score')
    ax.set_ylabel('Predicted Score')
    ax.set_title(f'(C) {best_model}: Predicted vs Actual\n'
                 f'PCC={pcc_best:.4f}, RMSE={rmse_best:.4f}', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (D) Purpose-based score distribution ──────────────────────────────────
    ax = axes[1][1]
    test_w = compute_purpose_scores(test_df)
    purpose_score_cols = [f'score_{p.replace("/","_").replace(" ","_")}'
                          for p in PURPOSE_WEIGHTS]
    purpose_labels = list(PURPOSE_WEIGHTS.keys())
    data_box = [test_w[c].values for c in purpose_score_cols]
    bp = ax.boxplot(data_box, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.2),
                    boxprops=dict(linewidth=1.2))
    box_colors_list = ['#2c7bb6','#74c476','#fd8d3c','#756bb1']
    for patch, color in zip(bp['boxes'], box_colors_list):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticklabels(purpose_labels, fontsize=9, rotation=10)
    ax.set_ylabel('Weighted Score (0-10)')
    ax.set_title('(D) Purpose-based Score Distribution\n(Same learners, different weights)',
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = Path(save_dir) / 'baseline_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n    Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. 결과 저장
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    results_df : pd.DataFrame,
    imp_df     : pd.DataFrame,
    save_dir   : str = '.',
) -> None:
    """CSV + JSON으로 결과 저장."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(save_dir / 'model_results.csv', index=False)
    imp_df.to_csv(save_dir / 'feature_importance.csv', index=False)

    # 요약 JSON (교수님께 보여주기 좋은 포맷)
    summary = {
        'dataset'      : 'speechocean762',
        'n_train'      : 2500,
        'n_test'       : 2500,
        'n_features'   : len(FEATURE_COLS),
        'feature_cols' : FEATURE_COLS,
        'models'       : results_df[['Model','Test_PCC','Test_RMSE','Test_MAE','Test_R2']]
                                   .to_dict(orient='records'),
        'purpose_weights': PURPOSE_WEIGHTS,
    }
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"    Saved: model_results.csv, feature_importance.csv, summary.json")


# ══════════════════════════════════════════════════════════════════════════════
# 9. 메인 실행
# ══════════════════════════════════════════════════════════════════════════════

def main():
    SAVE_DIR = './results'

    print("=" * 62)
    print("Baseline ML Model — Language Proficiency Assessment")
    print("Based on: Bamdev et al. (2023), IJAIED")
    print("=" * 62)

    # 1) 데이터 로드
    train_df, test_df = load_speechocean(verbose=True)

    # 2) 학습 & 평가
    results_df, models, (X_train, y_train, X_test, y_test) = \
        train_and_evaluate(train_df, test_df, verbose=True)

    # 베스트 모델 선택 (Test PCC 기준)
    best_model = results_df.loc[results_df['Test_PCC'].idxmax(), 'Model']
    best_pcc   = results_df['Test_PCC'].max()
    print(f"\n    Best model: {best_model} (PCC={best_pcc:.4f})")

    # 3) 피처 중요도 분석
    imp_df = feature_importance_analysis(models, X_train, y_train, verbose=True)

    # 4) 시각화 & 저장
    print("\n[4/4] Saving results...")
    Path(SAVE_DIR).mkdir(exist_ok=True)
    plot_all(results_df, imp_df, test_df, best_model,
             models, X_test, y_test, save_dir=SAVE_DIR)
    save_results(results_df, imp_df, save_dir=SAVE_DIR)

    # 5) 최종 요약 출력
    print("\n" + "=" * 62)
    print("FINAL RESULTS SUMMARY")
    print("=" * 62)
    print(f"\n{'Model':<22} {'PCC':>6} {'RMSE':>7} {'MAE':>6} {'R2':>7}")
    print("-" * 48)
    for _, row in results_df.sort_values('Test_PCC', ascending=False).iterrows():
        marker = " ← BEST" if row['Model'] == best_model else ""
        print(f"{row['Model']:<22} {row['Test_PCC']:>6.4f} "
              f"{row['Test_RMSE']:>7.4f} {row['Test_MAE']:>6.4f} "
              f"{row['Test_R2']:>7.4f}{marker}")

    print(f"\nTop-5 Important Features (RandomForest):")
    for _, row in imp_df.head(5).iterrows():
        bar = '█' * int(row['RF_Importance'] * 40)
        print(f"  {row['Feature']:<22} {row['RF_Importance']:.4f}  {bar}")

    print("\nPurpose-based Score Means (test set):")
    test_w = compute_purpose_scores(test_df)
    for purpose in PURPOSE_WEIGHTS:
        col  = f'score_{purpose.replace("/","_").replace(" ","_")}'
        mean = test_w[col].mean()
        std  = test_w[col].std()
        print(f"  {purpose:<22}: {mean:.3f} ± {std:.3f}")

    print("\nAll outputs saved to ./results/")
    print("  baseline_results.png   — visualization")
    print("  model_results.csv      — model metrics")
    print("  feature_importance.csv — feature importance")
    print("  summary.json           — full summary")


if __name__ == '__main__':
    main()
