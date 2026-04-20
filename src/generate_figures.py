"""
generate_figures.py
===================
논문용 Figure 전체 생성 스크립트.

ablation_study.py 실행 후 생성된 results/ 파일들을 기반으로
논문에 들어갈 Figure 4개를 한 번에 생성.

Figure 1: 가중치 수렴 궤적 (travel 목적, 에폭별)
Figure 2: Ablation 비교 막대그래프
Figure 3: CEFR 혼동행렬
Figure 4: 목적별 학습 가중치 히트맵 (최종)

실행 방법:
  python src/generate_figures.py
  python src/generate_figures.py --epochs 60  # 재학습 후 생성
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from dataset import ProficiencyDataset, load_speechocean
from model import DualHeadModelV2, UncertaintyWeighting
from trainer import evaluate

SEED     = 42
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PURPOSES = ['travel', 'business', 'academic']
SENT_F   = ['accuracy', 'completeness', 'fluency', 'prosodic']
CEFR_LABELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

PRESET_WEIGHTS = {
    'travel'  : [0.40, 0.10, 0.30, 0.20],
    'business': [0.30, 0.40, 0.20, 0.10],
    'academic': [0.20, 0.30, 0.20, 0.30],
}

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# ══════════════════════════════════════════════════════════════
# 학습 + 에폭별 가중치 기록
# ══════════════════════════════════════════════════════════════

def train_with_trajectory(cfg, train_ds, test_ds, device, epochs):
    """
    학습하면서 매 에폭마다 PurposeWeightGenerator 가중치를 기록.
    이게 논문 Figure 1 (수렴 궤적)의 데이터가 됨.
    """
    loader_tr = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    loader_te = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    model = DualHeadModelV2(cfg).to(device)
    uw    = UncertaintyWeighting(2).to(device)
    opt   = optim.AdamW(list(model.parameters()) + list(uw.parameters()),
                        lr=1e-3, weight_decay=1e-4)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gc    = nn.MSELoss()
    cc    = nn.CrossEntropyLoss()

    trajectory = []   # 에폭별 가중치 기록
    metrics_history = []

    for ep in range(1, epochs + 1):
        model.train()
        for batch in loader_tr:
            feat = batch['features'].to(device)
            pidx = batch['purpose_idx'].to(device)
            gt   = batch['goal_score'].to(device)
            ct   = batch['cefr_level'].to(device)
            sent = feat[:, [0, 3, 2, 1]]
            opt.zero_grad()
            gp, cl, _ = model(feat, pidx, sent)
            loss, _   = uw((gc(gp, gt), cc(cl, ct)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        # 에폭별 가중치 기록
        model.eval()
        with torch.no_grad():
            ep_weights = {'epoch': ep}
            for p, idx in [('travel',0),('business',1),('academic',2)]:
                pidx_t = torch.tensor([idx], dtype=torch.long).to(device)
                w = model.weight_generator(pidx_t).squeeze().cpu().numpy()
                for f, v in zip(SENT_F, w):
                    ep_weights[f'{p}_{f}'] = round(float(v), 4)
            trajectory.append(ep_weights)

        # 평가 지표 기록
        m = evaluate(model, loader_te, device)
        metrics_history.append({'epoch': ep, **m})

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:>3}/{epochs} | "
                  f"PCC={m['test/goal_pcc']:.4f} "
                  f"RMSE={m['test/goal_rmse']:.4f} "
                  f"CEFR={m['test/cefr_acc']:.4f}")

    return model, trajectory, metrics_history


# ══════════════════════════════════════════════════════════════
# Figure 1: 가중치 수렴 궤적
# ══════════════════════════════════════════════════════════════

def plot_weight_convergence(trajectory, save_dir):
    """
    논문 Figure 1:
    travel 목적의 4가지 피처 가중치가 에폭에 따라 수렴하는 궤적.
    문헌 기반 사전 설정값(점선)과 비교.
    """
    df   = pd.DataFrame(trajectory)
    eps  = df['epoch'].values
    colors = {'accuracy':'#4C72B0', 'completeness':'#C44E52',
               'fluency':'#55A868',  'prosodic':'#8172B2'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Figure 1: Purpose Weight Convergence Trajectory\n'
                 '(dashed = literature-based preset)',
                 fontsize=12, fontweight='bold')

    for ax, purpose in zip(axes, PURPOSES):
        preset = PRESET_WEIGHTS[purpose]
        for feat, color in colors.items():
            vals   = df[f'{purpose}_{feat}'].values
            preset_val = preset[SENT_F.index(feat)]
            ax.plot(eps, vals, color=color, lw=2, label=feat)
            ax.axhline(preset_val, color=color, lw=1.2,
                       ls='--', alpha=0.5)

        ax.set_title(f'{purpose.capitalize()} Purpose',
                     fontweight='bold', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Feature Weight')
        ax.set_ylim(-0.02, 0.72)
        ax.legend(fontsize=8, title='Feature', title_fontsize=8)
        ax.grid(alpha=0.3)

        # 시작점과 끝점 표시
        for feat, color in colors.items():
            vals = df[f'{purpose}_{feat}'].values
            ax.scatter([1], [vals[0]],  color=color, s=40, zorder=5)
            ax.scatter([eps[-1]], [vals[-1]], color=color, s=60,
                       marker='*', zorder=5)

    plt.tight_layout()
    out = Path(save_dir) / 'figure1_weight_convergence.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Save] {out}")


# ══════════════════════════════════════════════════════════════
# Figure 2: Ablation 비교 막대그래프
# ══════════════════════════════════════════════════════════════

def plot_ablation_bars(save_dir):
    """
    논문 Figure 2:
    각 모델의 RMSE와 CEFR Acc를 막대그래프로 비교.
    """
    csv_path = Path(save_dir) / 'ablation_results.csv'
    if not csv_path.exists():
        print(f"[Skip] {csv_path} not found. Run ablation_study.py first.")
        return

    df = pd.read_csv(csv_path)

    # CEFR N/A 처리
    df['CEFR Acc'] = pd.to_numeric(df['CEFR Acc'], errors='coerce')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 2: Ablation Study Results',
                 fontsize=13, fontweight='bold')

    models = df['Experiment'].tolist()
    short_names = df['Experiment'].apply(
        lambda x: x
        .replace('Baseline 1: Ridge', 'Ridge\nBaseline')
        .replace('Baseline 2: Single-Head MLP', 'Single-Head\nMLP')
        .replace('Ablation A: Fixed Loss Weight', 'Ablation A\n(Fixed LW)')
        .replace('Ablation B: Fixed Purpose Weight', 'Ablation B\n(Fixed PW)')
        .replace('Ablation A+B: Both Fixed', 'Ablation\nA+B')
        .replace('Ours: Full DualHead v2', 'Ours\n(Full)')
    ).tolist()
    n = len(short_names)
    base_colors = ['#95a5a6','#e67e22','#e67e22','#e67e22','#2ecc71']
    colors = base_colors[-n:]  # Ridge 없으면 5개, 있으면 6개 자동 대응

    # RMSE
    ax = axes[0]
    rmse_vals = df['Goal RMSE'].values
    bars = ax.bar(range(len(short_names)), rmse_vals,
                  color=colors[-len(short_names):], alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold' if val == min(rmse_vals) else 'normal')
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('(A) Goal Score RMSE', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # CEFR Acc
    ax = axes[1]
    cefr_vals = df['CEFR Acc'].values
    bars = ax.bar(range(len(short_names)), cefr_vals,
                  color=colors[-len(short_names):], alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, cefr_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2,
                    0.01, 'N/A', ha='center', va='bottom',
                    fontsize=9, color='gray')
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel('CEFR Exact Accuracy (higher is better)')
    ax.set_title('(B) CEFR Classification Accuracy', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # 범례
    legend_elements = [
        mpatches.Patch(color='#95a5a6', label='Baseline models'),
        mpatches.Patch(color='#e67e22', label='Ablation variants'),
        mpatches.Patch(color='#2ecc71', label='Our full model'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out = Path(save_dir) / 'figure2_ablation_bars.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Save] {out}")


# ══════════════════════════════════════════════════════════════
# Figure 3: CEFR 혼동행렬
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def plot_cefr_confusion(model, test_ds, device, save_dir):
    """논문 Figure 3: CEFR 혼동행렬."""
    model.eval()
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    c_pred, c_true = [], []

    for batch in loader:
        feat = batch['features'].to(device)
        pidx = batch['purpose_idx'].to(device)
        sent = feat[:, [0, 3, 2, 1]]
        _, cl, _ = model(feat, pidx, sent)
        c_pred.extend(cl.argmax(-1).cpu().numpy())
        c_true.extend(batch['cefr_level'].numpy())

    conf = np.zeros((6, 6), dtype=int)
    for t, p in zip(c_true, c_pred):
        conf[t][p] += 1

    # 정규화 (행 기준 %)
    conf_norm = conf.astype(float)
    row_sums = conf_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_norm = conf_norm / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 3: CEFR Level Classification\n'
                 f'Exact Acc={np.diag(conf).sum()/conf.sum():.4f}, '
                 f'±1 Acc={(np.abs(np.array(c_pred)-np.array(c_true))<=1).mean():.4f}',
                 fontsize=12, fontweight='bold')

    # 좌: 원본 count
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues',
                xticklabels=CEFR_LABELS, yticklabels=CEFR_LABELS,
                ax=axes[0], linewidths=0.5)
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    axes[0].set_title('(A) Count', fontweight='bold')

    # 우: 정규화 %
    sns.heatmap(conf_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CEFR_LABELS, yticklabels=CEFR_LABELS,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    axes[1].set_title('(B) Row-normalized', fontweight='bold')

    plt.tight_layout()
    out = Path(save_dir) / 'figure3_cefr_confusion.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Save] {out}")


# ══════════════════════════════════════════════════════════════
# Figure 4: 최종 가중치 히트맵
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def plot_weight_heatmap(model, device, save_dir):
    """논문 Figure 4: 학습된 vs 문헌 기반 가중치 히트맵."""
    model.eval()
    learned = []
    for idx in range(3):
        pidx = torch.tensor([idx], dtype=torch.long).to(device)
        w    = model.weight_generator(pidx).squeeze().cpu().numpy()
        learned.append(w)
    learned = np.array(learned)

    preset = np.array([PRESET_WEIGHTS[p] for p in PURPOSES])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Figure 4: Learned vs Literature-Based Purpose Weights',
                 fontsize=12, fontweight='bold')

    for ax, mat, title in zip(
        axes,
        [learned, preset],
        ['(A) Learned by Model', '(B) Literature Preset (AHP-based)']
    ):
        sns.heatmap(mat, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=SENT_F, yticklabels=PURPOSES,
                    ax=ax, vmin=0, vmax=0.6, annot_kws={'size': 12})
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Purpose')

    plt.tight_layout()
    out = Path(save_dir) / 'figure4_weight_heatmap.png'
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
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['training']['epochs'] = args.epochs
    cfg['wandb']['enabled']   = False

    Path(args.save_dir).mkdir(exist_ok=True)
    set_seed(SEED)

    print("=" * 62)
    print("Generate Paper Figures")
    print(f"Epochs: {args.epochs} | Device: {DEVICE}")
    print("=" * 62)

    # 데이터
    train_df, test_df = load_speechocean(cfg, verbose=False)
    train_ds = ProficiencyDataset(
        train_df, cfg['data']['feature_cols'], fit_scaler=True
    )
    test_ds = ProficiencyDataset(
        test_df, cfg['data']['feature_cols'], scaler=train_ds.scaler
    )

    # 학습 + 궤적 수집
    print(f"\n[Training {args.epochs} epochs + recording trajectory...]")
    model, trajectory, metrics_hist = train_with_trajectory(
        cfg, train_ds, test_ds, DEVICE, args.epochs
    )

    # Figure 생성
    print("\n[Generating figures...]")
    plot_weight_convergence(trajectory, args.save_dir)
    plot_ablation_bars(args.save_dir)
    plot_cefr_confusion(model, test_ds, DEVICE, args.save_dir)
    plot_weight_heatmap(model, DEVICE, args.save_dir)

    # 궤적 CSV 저장
    pd.DataFrame(trajectory).to_csv(
        f'{args.save_dir}/weight_trajectory.csv', index=False
    )

    print("\n" + "=" * 62)
    print("All figures saved:")
    print(f"  {args.save_dir}/figure1_weight_convergence.png")
    print(f"  {args.save_dir}/figure2_ablation_bars.png")
    print(f"  {args.save_dir}/figure3_cefr_confusion.png")
    print(f"  {args.save_dir}/figure4_weight_heatmap.png")
    print("=" * 62)


if __name__ == '__main__':
    main()
