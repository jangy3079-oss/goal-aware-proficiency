"""
robustness_test.py
==================
수식 암기 문제 검증 실험.

핵심 아이디어:
  - 학습 시: academic = [0.20, 0.30, 0.20, 0.30] 가중치로 goal_score 타겟 생성
  - 테스트 시: academic = [0.15, 0.35, 0.15, 0.35] (살짝 다른 가중치)로 타겟 생성

  만약 모델이 단순히 수식을 암기했다면:
    → 학습 가중치로 평가할 때 PCC가 높고
    → 다른 가중치로 평가할 때 PCC가 급격히 떨어짐

  만약 모델이 진짜 패턴을 학습했다면:
    → 두 경우 모두 비슷한 PCC 유지

실행 방법:
  python src/robustness_test.py
  python src/robustness_test.py --epochs 60

출력:
  results/robustness_results.csv
  results/robustness_plot.png
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
from scipy import stats as scipy_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from model import DualHeadModelV2, UncertaintyWeighting

SEED    = 42
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PURPOSES= ['travel', 'business', 'academic']
P2IDX   = {p: i for i, p in enumerate(PURPOSES)}

FEAT_COLS = [
    'accuracy', 'prosodic', 'fluency', 'completeness',
    'word_acc_mean', 'word_acc_std', 'word_acc_min', 'word_total_mean',
    'phone_acc_mean', 'phone_acc_std', 'phone_acc_min',
]
SENT_FEATS = ['accuracy', 'completeness', 'fluency', 'prosodic']

# ── 학습용 가중치 (원래 설정) ──────────────────────────────────
TRAIN_WEIGHTS = {
    'travel'  : {'accuracy': 0.40, 'completeness': 0.10,
                 'fluency' : 0.30, 'prosodic'    : 0.20},
    'business': {'accuracy': 0.30, 'completeness': 0.40,
                 'fluency' : 0.20, 'prosodic'    : 0.10},
    'academic': {'accuracy': 0.20, 'completeness': 0.30,
                 'fluency' : 0.20, 'prosodic'    : 0.30},
}

# ── 테스트용 가중치 3가지 케이스 ──────────────────────────────
# academic만 변경 (travel/business는 동일 유지)
# 케이스가 클수록 이동 폭이 커짐 → 더 강한 robustness 증거

# Case 1: 소폭 이동 (±0.05)
TEST_WEIGHTS_CASE1 = {
    'travel'  : {'accuracy': 0.40, 'completeness': 0.10,
                 'fluency' : 0.30, 'prosodic'    : 0.20},
    'business': {'accuracy': 0.30, 'completeness': 0.40,
                 'fluency' : 0.20, 'prosodic'    : 0.10},
    'academic': {'accuracy': 0.15, 'completeness': 0.35,   # ±0.05
                 'fluency' : 0.15, 'prosodic'    : 0.35},
}

# Case 2: 중폭 이동 — accuracy와 completeness 역전
# 학습: accuracy=0.20(낮음), completeness=0.30(중간)
# 테스트: accuracy=0.30(높음), completeness=0.15(낮음) → 우선순위 반전
TEST_WEIGHTS_CASE2 = {
    'travel'  : {'accuracy': 0.40, 'completeness': 0.10,
                 'fluency' : 0.30, 'prosodic'    : 0.20},
    'business': {'accuracy': 0.30, 'completeness': 0.40,
                 'fluency' : 0.20, 'prosodic'    : 0.10},
    'academic': {'accuracy': 0.30, 'completeness': 0.15,   # 역전
                 'fluency' : 0.30, 'prosodic'    : 0.25},
}

# Case 3: 극단적 이동 — completeness에 절반 몰기
# 학습: completeness=0.30 / 테스트: completeness=0.50 → 극단적 편중
TEST_WEIGHTS_CASE3 = {
    'travel'  : {'accuracy': 0.40, 'completeness': 0.10,
                 'fluency' : 0.30, 'prosodic'    : 0.20},
    'business': {'accuracy': 0.30, 'completeness': 0.40,
                 'fluency' : 0.20, 'prosodic'    : 0.10},
    'academic': {'accuracy': 0.10, 'completeness': 0.50,   # completeness에 절반
                 'fluency' : 0.10, 'prosodic'    : 0.30},
}

# 하위 호환성 유지
TEST_WEIGHTS_SHIFTED = TEST_WEIGHTS_CASE1

ALL_TEST_CASES = {
    'Case 1 (±0.05)' : TEST_WEIGHTS_CASE1,
    'Case 2 (reversed)' : TEST_WEIGHTS_CASE2,
    'Case 3 (extreme)': TEST_WEIGHTS_CASE3,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_to_cefr(score: float) -> int:
    thresholds = [2.0, 4.0, 6.0, 7.5, 9.0]
    for level, t in enumerate(thresholds):
        if score <= t:
            return level
    return 5


# ── Dataset ───────────────────────────────────────────────────
class RobustnessDataset(Dataset):
    """
    purpose_weights 파라미터로 goal_score 타겟을 다르게 생성.
    학습 시: TRAIN_WEIGHTS, 테스트 시: TEST_WEIGHTS_SHIFTED
    """
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler,
                 purpose_weights: dict, fit_scaler: bool = False):
        X = df[FEAT_COLS].values.astype(np.float32)
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X)

        # goal_score를 purpose_weights로 계산
        goal_scores = []
        for _, row in df.iterrows():
            p = row['purpose']
            w = purpose_weights[p]
            gs = sum(row[f] * w[f] for f in SENT_FEATS)
            goal_scores.append(gs)

        self.X           = torch.tensor(X, dtype=torch.float32)
        self.purpose_idx = torch.tensor(df['purpose_idx'].values, dtype=torch.long)
        self.goal_score  = torch.tensor(goal_scores, dtype=torch.float32)
        self.cefr_level  = torch.tensor(df['cefr_level'].values, dtype=torch.long)
        self.total_score = torch.tensor(df['total'].values, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return {
            'features'   : self.X[idx],
            'purpose_idx': self.purpose_idx[idx],
            'goal_score' : self.goal_score[idx],
            'cefr_level' : self.cefr_level[idx],
            'total_score': self.total_score[idx],
        }


def load_base_data():
    """speechocean762 로드 → 목적별 확장."""
    print("[Data] Loading speechocean762...")
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
            })
        return pd.DataFrame(records)

    def expand(df):
        rows = []
        for _, row in df.iterrows():
            cefr = score_to_cefr(row['total'])
            for p in PURPOSES:
                r = row.to_dict()
                r['purpose']     = p
                r['purpose_idx'] = P2IDX[p]
                r['cefr_level']  = cefr
                rows.append(r)
        return pd.DataFrame(rows)

    return expand(to_df(dataset['train'])), expand(to_df(dataset['test']))


def train_model(cfg, train_ds, device, epochs):
    """모델 학습 (TRAIN_WEIGHTS 기반 타겟으로)."""
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    model  = DualHeadModelV2(cfg).to(device)
    uw     = UncertaintyWeighting(2).to(device)
    opt    = optim.AdamW(list(model.parameters()) + list(uw.parameters()),
                         lr=1e-3, weight_decay=1e-4)
    sch    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gc     = nn.MSELoss()
    cc     = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs + 1):
        for batch in loader:
            feat  = batch['features'].to(device)
            pidx  = batch['purpose_idx'].to(device)
            gt    = batch['goal_score'].to(device)
            ct    = batch['cefr_level'].to(device)
            sent  = feat[:, [0, 3, 2, 1]]
            opt.zero_grad()
            gp, cl, _ = model(feat, pidx, sent)
            loss, _   = uw((gc(gp, gt), cc(cl, ct)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model


@torch.no_grad()
def evaluate_model(model, loader, device) -> Dict:
    """평가."""
    model.eval()
    gp, gt, cp, ct = [], [], [], []
    for batch in loader:
        feat = batch['features'].to(device)
        pidx = batch['purpose_idx'].to(device)
        sent = feat[:, [0, 3, 2, 1]]
        g, c, _ = model(feat, pidx, sent)
        gp.extend(g.cpu().numpy())
        gt.extend(batch['goal_score'].numpy())
        cp.extend(c.argmax(-1).cpu().numpy())
        ct.extend(batch['cefr_level'].numpy())

    gp, gt = np.array(gp), np.array(gt)
    cp, ct = np.array(cp), np.array(ct)
    pcc, _ = scipy_stats.pearsonr(gt, gp)
    return {
        'PCC' : round(float(pcc),                        4),
        'RMSE': round(float(np.sqrt(mean_squared_error(gt, gp))), 4),
        'MAE' : round(float(mean_absolute_error(gt, gp)), 4),
        'CEFR': round(float((cp == ct).mean()),           4),
    }


@torch.no_grad()
def evaluate_by_purpose(model, test_df, scaler,
                         purpose_weights, device) -> Dict:
    """목적별 평가."""
    results = {}
    for purpose in PURPOSES:
        sub = test_df[test_df['purpose'] == purpose].copy()
        ds  = RobustnessDataset(sub, scaler, purpose_weights)
        ld  = DataLoader(ds, batch_size=256, shuffle=False)
        m   = evaluate_model(model, ld, device)
        results[purpose] = m
    return results


def plot_robustness_multi(results_orig: Dict,
                          all_case_results: Dict,
                          save_dir: str) -> None:
    """
    3가지 이동 케이스의 academic PCC/RMSE 변화를 한 번에 비교.
    논문 Figure용 — 이동 폭이 커져도 PCC가 유지됨을 시각화.
    """
    case_names = list(all_case_results.keys())
    orig_pcc   = results_orig['academic']['PCC']
    orig_rmse  = results_orig['academic']['RMSE']

    shifted_pccs  = [all_case_results[c]['academic']['PCC']  for c in case_names]
    shifted_rmses = [all_case_results[c]['academic']['RMSE'] for c in case_names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'Robustness Test: Academic Purpose under Shifted Test Weights\n'
        'Train weights fixed at [0.20, 0.30, 0.20, 0.30]',
        fontsize=11, fontweight='bold'
    )

    x      = np.arange(len(case_names))
    colors = ['#4C72B0', '#e67e22', '#C44E52']
    labels = [
        'Case 1: ±0.05\n[0.15,0.35,0.15,0.35]',
        'Case 2: Reversed\n[0.30,0.15,0.30,0.25]',
        'Case 3: Extreme\n[0.10,0.50,0.10,0.30]',
    ]

    for ax, orig_val, shifted_vals, metric, title, better in zip(
        axes,
        [orig_pcc,  orig_rmse],
        [shifted_pccs, shifted_rmses],
        ['PCC', 'RMSE'],
        ['(A) PCC — higher is better', '(B) RMSE — lower is better'],
        ['high', 'low']
    ):
        # 원래 값 점선
        ax.axhline(orig_val, color='green', lw=2, ls='--',
                   label=f'Original ({orig_val:.4f})', zorder=3)

        bars = ax.bar(x, shifted_vals, color=colors, alpha=0.85,
                      edgecolor='white', linewidth=0.5, width=0.5)

        for bar, val in zip(bars, shifted_vals):
            delta = val - orig_val
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.001 if metric == 'PCC' else 0.002),
                    f'{val:.4f}\n({delta:+.4f})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        if metric == 'PCC':
            ax.set_ylim(max(0, min(shifted_vals) - 0.05), 1.02)
        else:
            ax.set_ylim(0, max(shifted_vals) * 1.3)

    plt.tight_layout()
    out = Path(save_dir) / 'robustness_plot.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Save] {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   type=str, default='src/config.yaml')
    parser.add_argument('--epochs',   type=int, default=60)
    parser.add_argument('--save-dir', type=str, default='results')
    args = parser.parse_args()

    # config 로드
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['training']['epochs'] = args.epochs

    Path(args.save_dir).mkdir(exist_ok=True)
    set_seed(SEED)

    print("=" * 62)
    print("Robustness Test: Shifted Test Weights (3 Cases)")
    print(f"Epochs: {args.epochs} | Device: {DEVICE}")
    print("=" * 62)
    print("\n[Design]")
    print("  Train academic : [0.20, 0.30, 0.20, 0.30]  (fixed)")
    print("  Case 1 (±0.05) : [0.15, 0.35, 0.15, 0.35]  small shift")
    print("  Case 2 (reversed): [0.30, 0.15, 0.30, 0.25]  priority reversed")
    print("  Case 3 (extreme) : [0.10, 0.50, 0.10, 0.30]  completeness x5")
    print("  → If PCC drops sharply: model memorized the formula")
    print("  → If PCC stays stable : model learned real patterns")

    # 데이터 로드
    train_df, test_df = load_base_data()
    train_ds = RobustnessDataset(
        train_df, scaler=None,
        purpose_weights=TRAIN_WEIGHTS, fit_scaler=True
    )

    # 모델 학습 (한 번만)
    print(f"\n[Training] {args.epochs} epochs with TRAIN_WEIGHTS...")
    model = train_model(cfg, train_ds, DEVICE, args.epochs)
    print("  Training complete.")

    # 원래 가중치 평가
    print("\n[Eval 0] Original weights (same as train)...")
    by_purpose_orig = evaluate_by_purpose(
        model, test_df, train_ds.scaler, TRAIN_WEIGHTS, DEVICE
    )

    # 3가지 케이스 평가
    all_case_results = {}
    for case_name, case_weights in ALL_TEST_CASES.items():
        print(f"[Eval] {case_name}...")
        all_case_results[case_name] = evaluate_by_purpose(
            model, test_df, train_ds.scaler, case_weights, DEVICE
        )

    # 결과 출력
    print("\n" + "=" * 70)
    print("ROBUSTNESS TEST RESULTS — Academic Purpose PCC")
    print("=" * 70)
    print(f"\n  {'Case':<22} {'Original':>10} {'Shifted':>10} {'Δ PCC':>10} {'Verdict':>15}")
    print("  " + "-" * 70)

    orig_ac = by_purpose_orig['academic']['PCC']
    rows = []
    for case_name, case_res in all_case_results.items():
        shift_ac = case_res['academic']['PCC']
        delta    = shift_ac - orig_ac
        verdict  = "Robust ✓" if abs(delta) < 0.02 else \
                   "Partial"  if abs(delta) < 0.05 else \
                   "Memorized ✗"
        print(f"  {case_name:<22} {orig_ac:>10.4f} {shift_ac:>10.4f} "
              f"{delta:>+10.4f} {verdict:>15}")
        rows.append({
            'Case'        : case_name,
            'PCC_original': orig_ac,
            'PCC_shifted' : shift_ac,
            'Delta_PCC'   : round(delta, 4),
            'RMSE_original': by_purpose_orig['academic']['RMSE'],
            'RMSE_shifted' : case_res['academic']['RMSE'],
            'Verdict'     : verdict,
        })

    # 핵심 해석
    max_drop = max(abs(r['Delta_PCC']) for r in rows)
    print(f"\n  [핵심 해석]")
    print(f"  최대 PCC 변화량: {max_drop:.4f} (Case 3 극단적 이동 기준)")
    if max_drop < 0.02:
        print("  → 모든 케이스에서 Robust: 모델이 공식 암기가 아닌 패턴을 학습")
        print("  → 논문 robustness 증거로 강하게 사용 가능")
    elif max_drop < 0.05:
        print("  → 부분적 Robust: 소폭~중폭 이동에서는 안정적")
        print("  → Limitation에서 극단적 케이스 언급 필요")
    else:
        print("  → 극단적 이동에서 PCC 감소: 대리 레이블 의존성 존재")
        print("  → Limitation에서 솔직하게 논의 필요")

    # 저장
    df_res = pd.DataFrame(rows)
    df_res.to_csv(f'{args.save_dir}/robustness_results.csv', index=False)
    print(f"\n[Save] {args.save_dir}/robustness_results.csv")

    # 시각화 (3케이스 비교)
    plot_robustness_multi(by_purpose_orig, all_case_results, args.save_dir)
    print("\nDone.")


if __name__ == '__main__':
    main()
