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

# ── 테스트용 가중치 (의도적으로 다르게 설정) ──────────────────
# academic만 변경: completeness ↑, accuracy ↓, fluency ↓
# 모델이 암기했다면 이 가중치로 평가할 때 PCC가 크게 떨어져야 함
TEST_WEIGHTS_SHIFTED = {
    'travel'  : {'accuracy': 0.40, 'completeness': 0.10,   # 동일
                 'fluency' : 0.30, 'prosodic'    : 0.20},
    'business': {'accuracy': 0.30, 'completeness': 0.40,   # 동일
                 'fluency' : 0.20, 'prosodic'    : 0.10},
    'academic': {'accuracy': 0.15, 'completeness': 0.35,   # ← 변경
                 'fluency' : 0.15, 'prosodic'    : 0.35},
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


def plot_robustness(results_orig: Dict, results_shift: Dict,
                    save_dir: str) -> None:
    """
    원래 가중치 vs 변경된 가중치로 평가한 결과 비교.
    목적별로 PCC, RMSE를 나란히 보여줌.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        'Robustness Test: Train Weights vs Shifted Test Weights\n'
        '(Academic purpose only shifted: [0.20,0.30,0.20,0.30] → [0.15,0.35,0.15,0.35])',
        fontsize=11, fontweight='bold'
    )

    x      = np.arange(len(PURPOSES))
    width  = 0.35
    colors = ['#4C72B0', '#C44E52']

    for ax, metric, title in zip(
        axes,
        ['PCC', 'RMSE'],
        ['(A) PCC — higher is better', '(B) RMSE — lower is better']
    ):
        orig  = [results_orig[p][metric]  for p in PURPOSES]
        shift = [results_shift[p][metric] for p in PURPOSES]

        b1 = ax.bar(x - width/2, orig,  width, label='Original weights',
                    color=colors[0], alpha=0.85)
        b2 = ax.bar(x + width/2, shift, width, label='Shifted weights (academic)',
                    color=colors[1], alpha=0.85)

        for bar, val in zip(b1, orig):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(b2, shift):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(PURPOSES, fontsize=11)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    # academic 차이 강조 화살표
    ax = axes[0]
    orig_ac  = results_orig['academic']['PCC']
    shift_ac = results_shift['academic']['PCC']
    diff     = orig_ac - shift_ac
    ax.annotate(
        f'Δ={diff:+.4f}',
        xy=(2 + width/2, shift_ac),
        xytext=(2 + width/2 + 0.3, shift_ac - 0.01),
        fontsize=9, color='red', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='red')
    )

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
    print("Robustness Test: Shifted Test Weights")
    print(f"Epochs: {args.epochs} | Device: {DEVICE}")
    print("=" * 62)
    print("\n[Design]")
    print("  Train academic: [0.20, 0.30, 0.20, 0.30]")
    print("  Test  academic: [0.15, 0.35, 0.15, 0.35]  ← shifted")
    print("  If PCC drops sharply → model memorized the formula")
    print("  If PCC stays stable  → model learned real patterns")

    # 데이터 로드
    train_df, test_df = load_base_data()

    # 학습 데이터셋 (TRAIN_WEIGHTS)
    train_ds = RobustnessDataset(
        train_df, scaler=None,
        purpose_weights=TRAIN_WEIGHTS, fit_scaler=True
    )

    # 모델 학습
    print(f"\n[Training] {args.epochs} epochs with TRAIN_WEIGHTS...")
    model = train_model(cfg, train_ds, DEVICE, args.epochs)
    print("  Training complete.")

    # 평가 1: 원래 가중치로 평가
    print("\n[Eval 1] Original test weights (same as train)...")
    test_ds_orig = RobustnessDataset(
        test_df, scaler=train_ds.scaler,
        purpose_weights=TRAIN_WEIGHTS
    )
    ld_orig = DataLoader(test_ds_orig, batch_size=256, shuffle=False)
    overall_orig = evaluate_model(model, ld_orig, DEVICE)
    by_purpose_orig = evaluate_by_purpose(
        model, test_df, train_ds.scaler, TRAIN_WEIGHTS, DEVICE
    )

    # 평가 2: 변경된 가중치로 평가
    print("[Eval 2] Shifted test weights (academic only shifted)...")
    test_ds_shift = RobustnessDataset(
        test_df, scaler=train_ds.scaler,
        purpose_weights=TEST_WEIGHTS_SHIFTED
    )
    ld_shift = DataLoader(test_ds_shift, batch_size=256, shuffle=False)
    overall_shift = evaluate_model(model, ld_shift, DEVICE)
    by_purpose_shift = evaluate_by_purpose(
        model, test_df, train_ds.scaler, TEST_WEIGHTS_SHIFTED, DEVICE
    )

    # 결과 출력
    print("\n" + "=" * 62)
    print("ROBUSTNESS TEST RESULTS")
    print("=" * 62)

    print(f"\n  Overall:")
    print(f"  {'Metric':<8} {'Original':>12} {'Shifted':>12} {'Δ':>10}")
    print("  " + "-" * 44)
    for m in ['PCC', 'RMSE', 'MAE', 'CEFR']:
        o = overall_orig[m]
        s = overall_shift[m]
        print(f"  {m:<8} {o:>12.4f} {s:>12.4f} {s-o:>+10.4f}")

    print(f"\n  By Purpose (PCC):")
    print(f"  {'Purpose':<12} {'Original':>10} {'Shifted':>10} {'Δ':>10} {'Verdict':>20}")
    print("  " + "-" * 62)
    for p in PURPOSES:
        o = by_purpose_orig[p]['PCC']
        s = by_purpose_shift[p]['PCC']
        d = s - o
        shifted = "(shifted)" if p == 'academic' else ""
        verdict = "Memorized?" if abs(d) > 0.05 else "Robust"
        print(f"  {p:<12} {o:>10.4f} {s:>10.4f} {d:>+10.4f} "
              f"{verdict:>15} {shifted}")

    # 핵심 해석
    academic_drop = by_purpose_orig['academic']['PCC'] - \
                    by_purpose_shift['academic']['PCC']
    print(f"\n  [핵심 해석]")
    if abs(academic_drop) < 0.02:
        print(f"  Academic PCC 변화: {academic_drop:+.4f} → 거의 없음")
        print(f"  → 모델이 단순 수식 암기가 아닌 실제 패턴을 학습했을 가능성")
        print(f"  → 논문에서 robustness 증거로 사용 가능")
    elif abs(academic_drop) < 0.05:
        print(f"  Academic PCC 변화: {academic_drop:+.4f} → 소폭 감소")
        print(f"  → 부분적 암기 + 부분적 패턴 학습 혼재")
        print(f"  → Limitation에서 솔직하게 언급 필요")
    else:
        print(f"  Academic PCC 변화: {academic_drop:+.4f} → 큰 폭 감소")
        print(f"  → 수식 암기 가능성 높음")
        print(f"  → Limitation에서 명확히 언급 필요")

    # 저장
    rows = []
    for p in PURPOSES:
        o = by_purpose_orig[p]
        s = by_purpose_shift[p]
        rows.append({
            'Purpose'     : p,
            'PCC_original': o['PCC'],  'PCC_shifted': s['PCC'],
            'RMSE_original': o['RMSE'],'RMSE_shifted': s['RMSE'],
            'Delta_PCC'   : round(s['PCC'] - o['PCC'], 4),
        })
    df_res = pd.DataFrame(rows)
    df_res.to_csv(f'{args.save_dir}/robustness_results.csv', index=False)
    print(f"\n[Save] {args.save_dir}/robustness_results.csv")

    plot_robustness(by_purpose_orig, by_purpose_shift, args.save_dir)
    print("\nDone.")


if __name__ == '__main__':
    main()
