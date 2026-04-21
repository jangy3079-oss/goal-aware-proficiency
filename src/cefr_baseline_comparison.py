"""
cefr_baseline_comparison.py
===========================
CEFR Head의 실제 기여 검증 실험.

질문:
  "단순히 총점(total)을 예측하고 threshold로 CEFR을 구간 분류하는 것과
   CEFR Head가 직접 6-class 분류하는 것이 구체적으로 어떻게 다른가?"

실험 구성:
  Baseline C1 : Ridge 회귀로 total 예측 → threshold로 CEFR 변환
  Baseline C2 : XGBoost로 total 예측 → threshold로 CEFR 변환
  Baseline C3 : 피처 11개 → 직접 6-class SVM 분류 (CEFR을 분류 문제로)
  Baseline C4 : 피처 11개 → 직접 6-class Random Forest 분류
  Ours        : DualHead v2의 CEFR Head (공유 표현 기반 분류)

만약 Ours > Baseline C1~C4 → CEFR Head가 진짜 학습을 함
만약 Ours ≈ Baseline C1~C4 → CEFR Head는 threshold 분류와 다를 바 없음

실행 방법:
  python src/cefr_baseline_comparison.py
  python src/cefr_baseline_comparison.py --epochs 60
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from dataset import ProficiencyDataset, load_speechocean
from model import DualHeadModelV2, UncertaintyWeighting

SEED    = 42
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CEFR_LABELS  = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
THRESHOLDS   = [2.0, 4.0, 6.0, 7.5, 9.0]
FEAT_COLS    = [
    'accuracy', 'prosodic', 'fluency', 'completeness',
    'word_acc_mean', 'word_acc_std', 'word_acc_min', 'word_total_mean',
    'phone_acc_mean', 'phone_acc_std', 'phone_acc_min',
]

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def score_to_cefr(score: float) -> int:
    for level, t in enumerate(THRESHOLDS):
        if score <= t: return level
    return 5

def cefr_metrics(y_true, y_pred) -> Dict:
    """정확도 + ±1 정확도 + 매크로 F1 계산."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    exact  = float((y_true == y_pred).mean())
    pm1    = float((np.abs(y_true - y_pred) <= 1).mean())
    # 매크로 F1 (클래스 불균형 고려)
    from sklearn.metrics import f1_score
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    return {
        'Exact Acc': round(exact, 4),
        '±1 Acc'   : round(pm1,   4),
        'Macro F1' : round(f1,    4),
    }


# ══════════════════════════════════════════════════════════════
# 데이터 로드 (base: 목적 확장 없이 원본 2,500개)
# ══════════════════════════════════════════════════════════════

def load_base_data():
    """목적 확장 없이 원본 2,500개 로드."""
    print("[Data] Loading speechocean762 (base, no purpose expansion)...")
    dataset = load_dataset("mispeech/speechocean762")

    def to_df(split):
        sp = split.remove_columns(['audio']) \
            if 'audio' in split.column_names else split
        records = []
        for item in sp:
            word_acc  = [w['accuracy'] for w in item['words']]
            word_tot  = [w['total']    for w in item['words']]
            phone_acc = []
            for w in item['words']:
                phone_acc.extend(w['phones-accuracy'])
            records.append({
                'total'          : float(item['total']),
                'accuracy'       : float(item['accuracy']),
                'completeness'   : float(item['completeness']),
                'fluency'        : float(item['fluency']),
                'prosodic'       : float(item['prosodic']),
                'word_acc_mean'  : float(np.mean(word_acc)),
                'word_acc_std'   : float(np.std(word_acc)),
                'word_acc_min'   : float(np.min(word_acc)),
                'word_total_mean': float(np.mean(word_tot)),
                'phone_acc_mean' : float(np.mean(phone_acc)),
                'phone_acc_std'  : float(np.std(phone_acc)),
                'phone_acc_min'  : float(np.min(phone_acc)),
                'cefr_level'     : score_to_cefr(float(item['total'])),
            })
        return pd.DataFrame(records)

    return to_df(dataset['train']), to_df(dataset['test'])


# ══════════════════════════════════════════════════════════════
# Baseline C1~C4: 전통 ML 기반 CEFR 분류
# ══════════════════════════════════════════════════════════════

def run_ml_baselines(train_df, test_df) -> Dict:
    """
    4가지 ML baseline CEFR 분류.

    C1: Ridge → total 예측 → threshold 변환
    C2: XGBoost → total 예측 → threshold 변환
    C3: SVM 직접 6-class 분류
    C4: Random Forest 직접 6-class 분류
    """
    X_train = train_df[FEAT_COLS].values
    X_test  = test_df[FEAT_COLS].values
    y_train_total = train_df['total'].values
    y_test_cefr   = test_df['cefr_level'].values

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    results = {}

    # C1: Ridge → threshold
    print("  [C1] Ridge → threshold conversion...")
    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_tr_sc, y_train_total)
    total_pred = ridge.predict(X_te_sc)
    cefr_pred  = np.array([score_to_cefr(s) for s in total_pred])
    results['C1: Ridge→Threshold'] = cefr_metrics(y_test_cefr, cefr_pred)

    # C2: XGBoost → threshold
    print("  [C2] XGBoost → threshold conversion...")
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                           random_state=SEED, verbosity=0)
        xgb.fit(X_tr_sc, y_train_total)
        total_pred = xgb.predict(X_te_sc)
        cefr_pred  = np.array([score_to_cefr(s) for s in total_pred])
        results['C2: XGBoost→Threshold'] = cefr_metrics(y_test_cefr, cefr_pred)
    except ImportError:
        print("    [Skip] xgboost not installed")

    # C3: SVM 직접 분류
    print("  [C3] SVM direct 6-class classification...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
    svm.fit(X_tr_sc, train_df['cefr_level'].values)
    cefr_pred = svm.predict(X_te_sc)
    results['C3: SVM Direct'] = cefr_metrics(y_test_cefr, cefr_pred)

    # C4: Random Forest 직접 분류
    print("  [C4] Random Forest direct 6-class classification...")
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_sc, train_df['cefr_level'].values)
    cefr_pred = rf.predict(X_te_sc)
    results['C4: RF Direct'] = cefr_metrics(y_test_cefr, cefr_pred)

    return results


# ══════════════════════════════════════════════════════════════
# Ours: DualHead CEFR Head
# ══════════════════════════════════════════════════════════════

def run_dualhead_cefr(cfg, train_df, test_df, epochs) -> Dict:
    """DualHead v2 학습 후 CEFR Head 성능만 추출."""
    print(f"  [Ours] Training DualHead v2 ({epochs} epochs)...")

    # 목적 확장 버전 로드 (DualHead 학습용)
    from dataset import load_speechocean
    train_exp, test_exp = load_speechocean(cfg, verbose=False)

    train_ds = ProficiencyDataset(
        train_exp, cfg['data']['feature_cols'], fit_scaler=True
    )
    # test는 원본 base_df의 CEFR 레이블로 평가
    test_ds = ProficiencyDataset(
        test_exp, cfg['data']['feature_cols'], scaler=train_ds.scaler
    )

    train_ld = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    model = DualHeadModelV2(cfg).to(DEVICE)
    uw    = UncertaintyWeighting(2).to(DEVICE)
    opt   = optim.AdamW(
        list(model.parameters()) + list(uw.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gc, cc = nn.MSELoss(), nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs + 1):
        for batch in train_ld:
            feat = batch['features'].to(DEVICE)
            pidx = batch['purpose_idx'].to(DEVICE)
            gt   = batch['goal_score'].to(DEVICE)
            ct   = batch['cefr_level'].to(DEVICE)
            sent = feat[:, [0, 3, 2, 1]]
            opt.zero_grad()
            gp, cl, _ = model(feat, pidx, sent)
            loss, _   = uw((gc(gp, gt), cc(cl, ct)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        if ep % 20 == 0 or ep == epochs:
            print(f"    Ep {ep}/{epochs}")

    # CEFR 평가
    model.eval()
    c_pred, c_true = [], []
    with torch.no_grad():
        for batch in test_ld:
            feat = batch['features'].to(DEVICE)
            pidx = batch['purpose_idx'].to(DEVICE)
            sent = feat[:, [0, 3, 2, 1]]
            _, cl, _ = model(feat, pidx, sent)
            c_pred.extend(cl.argmax(-1).cpu().numpy())
            c_true.extend(batch['cefr_level'].numpy())

    return cefr_metrics(c_true, c_pred), model, train_ds.scaler


# ══════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════

def plot_comparison(results: Dict, save_dir: str) -> None:
    """CEFR 분류 성능 비교 막대그래프."""
    names   = list(results.keys())
    metrics = ['Exact Acc', '±1 Acc', 'Macro F1']
    colors_map = {
        'C1': '#95a5a6', 'C2': '#95a5a6',
        'C3': '#e67e22', 'C4': '#e67e22',
        'Ours': '#2ecc71',
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('CEFR Classification: DualHead Head vs. Baselines\n'
                 '(Does the CEFR Head learn beyond simple threshold conversion?)',
                 fontsize=11, fontweight='bold')

    for ax, metric in zip(axes, metrics):
        vals   = [results[n][metric] for n in names]
        colors = []
        for n in names:
            key = 'Ours' if 'Ours' in n else n[:2]
            colors.append(colors_map.get(key, '#95a5a6'))

        bars = ax.bar(range(len(names)), vals,
                      color=colors, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=9,
                    fontweight='bold' if val == max(vals) else 'normal')

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(
            [n.replace(': ', ':\n') for n in names],
            fontsize=8, rotation=0
        )
        ax.set_title(f'({chr(65+metrics.index(metric))}) {metric}',
                     fontweight='bold')
        ax.set_ylim(0, min(1.08, max(vals) * 1.1))
        ax.grid(axis='y', alpha=0.3)

    # 범례
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color='#95a5a6', label='Threshold-based (C1, C2)'),
        mpatches.Patch(color='#e67e22', label='Direct classification (C3, C4)'),
        mpatches.Patch(color='#2ecc71', label='Ours: DualHead CEFR Head'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out = Path(save_dir) / 'figure_cefr_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Save] {out}")


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   type=str, default='src/config.yaml')
    parser.add_argument('--epochs',   type=int, default=60)
    parser.add_argument('--save-dir', type=str, default='results')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg['training']['epochs']  = args.epochs
    cfg['wandb']['enabled']    = False

    Path(args.save_dir).mkdir(exist_ok=True)
    set_seed(SEED)

    print("=" * 62)
    print("CEFR Head Contribution Verification")
    print(f"Epochs: {args.epochs} | Device: {DEVICE}")
    print("=" * 62)
    print("\n[Question]")
    print("  Is the CEFR Head actually learning something meaningful,")
    print("  or is it equivalent to simple threshold conversion?")

    # 데이터 로드
    train_df, test_df = load_base_data()
    print(f"\n  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  CEFR distribution (test):")
    for lvl, cnt in pd.Series(test_df['cefr_level']).value_counts().sort_index().items():
        print(f"    {CEFR_LABELS[lvl]}: {cnt} ({cnt/len(test_df)*100:.1f}%)")

    # ML Baselines
    print("\n[ML Baselines]")
    ml_results = run_ml_baselines(train_df, test_df)

    # DualHead CEFR Head
    print("\n[DualHead CEFR Head]")
    our_metrics, model, scaler = run_dualhead_cefr(cfg, train_df, test_df, args.epochs)

    # 전체 결과 합치기
    all_results = {**ml_results, 'Ours: DualHead CEFR': our_metrics}

    # 결과 출력
    print("\n" + "=" * 62)
    print("CEFR CLASSIFICATION COMPARISON")
    print("=" * 62)
    print(f"\n  {'Model':<28} {'Exact Acc':>10} {'±1 Acc':>8} {'Macro F1':>10}")
    print("  " + "-" * 58)
    for name, m in all_results.items():
        marker = " ← Ours" if "Ours" in name else ""
        print(f"  {name:<28} {m['Exact Acc']:>10.4f} "
              f"{m['±1 Acc']:>8.4f} {m['Macro F1']:>10.4f}{marker}")

    # 핵심 해석
    our_exact = our_metrics['Exact Acc']
    our_f1    = our_metrics['Macro F1']
    best_ml_exact = max(m['Exact Acc'] for m in ml_results.values())
    best_ml_f1    = max(m['Macro F1']  for m in ml_results.values())

    print(f"\n  [핵심 해석]")
    print(f"  Exact Acc 차이: Ours({our_exact:.4f}) vs Best ML({best_ml_exact:.4f})"
          f" = {our_exact - best_ml_exact:+.4f}")
    print(f"  Macro F1  차이: Ours({our_f1:.4f}) vs Best ML({best_ml_f1:.4f})"
          f" = {our_f1 - best_ml_f1:+.4f}")

    if our_exact > best_ml_exact + 0.02:
        print("  → CEFR Head가 단순 threshold 변환보다 의미 있게 더 잘 학습함")
        print("  → 논문 contribution으로 강하게 주장 가능")
    elif our_exact > best_ml_exact:
        print("  → CEFR Head가 소폭 우위 — 기여는 있으나 미미한 수준")
        print("  → 수치보다 ±1 Acc와 Macro F1로 보완 필요")
    else:
        print("  → CEFR Head가 baseline 대비 우위 없음")
        print("  → CEFR Head를 주 contribution에서 빼고 goal score에 집중 권장")

    # 저장
    df_res = pd.DataFrame([
        {'Model': name, **m} for name, m in all_results.items()
    ])
    df_res.to_csv(f'{args.save_dir}/cefr_comparison.csv', index=False)
    print(f"\n[Save] {args.save_dir}/cefr_comparison.csv")

    plot_comparison(all_results, args.save_dir)
    print("\nDone.")


if __name__ == '__main__':
    main()
