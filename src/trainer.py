"""
trainer.py
==========
학습 루프 + WandB 로깅 + 평가 + 시각화.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats as scipy_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from model import DualHeadModelV2, UncertaintyWeighting

CEFR_LABELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
PURPOSES    = ['travel', 'business', 'academic']
SENT_FEATS  = ['accuracy', 'completeness', 'fluency', 'prosodic']


# ── 평가 지표 ───────────────────────────────────────────────────
def compute_metrics(
    g_pred: np.ndarray, g_true: np.ndarray,
    c_pred: np.ndarray, c_true: np.ndarray,
) -> Dict[str, float]:
    pcc, _  = scipy_stats.pearsonr(g_true, g_pred)
    mse     = mean_squared_error(g_true, g_pred)
    mae     = mean_absolute_error(g_true, g_pred)
    c_acc   = (c_pred == c_true).mean()
    c_pm1   = (np.abs(c_pred - c_true) <= 1).mean()
    return {
        'goal_pcc' : round(float(pcc),          4),
        'goal_rmse': round(float(np.sqrt(mse)), 4),
        'goal_mae' : round(float(mae),          4),
        'cefr_acc' : round(float(c_acc),        4),
        'cefr_pm1' : round(float(c_pm1),        4),
    }


# ── 한 에폭 학습 ────────────────────────────────────────────────
def train_one_epoch(
    model       : DualHeadModelV2,
    loader      : DataLoader,
    optimizer   : optim.Optimizer,
    goal_crit   : nn.Module,
    cefr_crit   : nn.Module,
    uw          : UncertaintyWeighting,
    device      : torch.device,
    use_uw      : bool = True,
    goal_w      : float = 0.6,
    cefr_w      : float = 0.4,
) -> Dict[str, float]:
    model.train()
    total_sum = goal_sum = cefr_sum = 0.0
    uw_weights = [goal_w, cefr_w]

    for batch in loader:
        feat   = batch['features'].to(device)
        pidx   = batch['purpose_idx'].to(device)
        gtarget= batch['goal_score'].to(device)
        ctarget= batch['cefr_level'].to(device)
        sent   = feat[:, [0, 3, 2, 1]]   # accuracy/completeness/fluency/prosodic

        optimizer.zero_grad()
        gp, cl, _ = model(feat, pidx, sent)

        g_loss = goal_crit(gp, gtarget)
        c_loss = cefr_crit(cl, ctarget)

        if use_uw:
            loss, uw_weights = uw((g_loss, c_loss))
        else:
            loss = goal_w * g_loss + cefr_w * c_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_sum += loss.item()
        goal_sum  += g_loss.item()
        cefr_sum  += c_loss.item()

    n = len(loader)
    return {
        'train/total_loss'  : total_sum / n,
        'train/goal_loss'   : goal_sum  / n,
        'train/cefr_loss'   : cefr_sum  / n,
        'train/goal_weight' : uw_weights[0],
        'train/cefr_weight' : uw_weights[1],
    }


# ── 평가 ────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model : DualHeadModelV2,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    g_pred, g_true, c_pred, c_true = [], [], [], []

    for batch in loader:
        feat = batch['features'].to(device)
        pidx = batch['purpose_idx'].to(device)
        sent = feat[:, [0, 3, 2, 1]]
        gp, cl, _ = model(feat, pidx, sent)
        g_pred.extend(gp.cpu().numpy())
        g_true.extend(batch['goal_score'].numpy())
        c_pred.extend(cl.argmax(-1).cpu().numpy())
        c_true.extend(batch['cefr_level'].numpy())

    metrics = compute_metrics(
        np.array(g_pred), np.array(g_true),
        np.array(c_pred), np.array(c_true),
    )
    return {f'test/{k}': v for k, v in metrics.items()}


# ── 학습된 가중치 히트맵 (WandB 로깅용) ──────────────────────────
@torch.no_grad()
def get_learned_weights(
    model : DualHeadModelV2,
    device: torch.device,
) -> np.ndarray:
    """
    (3, 4) 배열 반환: 목적별 × 피처별 학습 가중치.
    WandB에 히트맵으로 업로드 → 학습 중 가중치 변화 추이 확인 가능.
    """
    model.eval()
    weights = []
    for idx in range(len(PURPOSES)):
        pidx = torch.tensor([idx], dtype=torch.long).to(device)
        w    = model.weight_generator(pidx).squeeze().cpu().numpy()
        weights.append(w)
    return np.array(weights)   # (3, 4)


# ── WandB 히트맵 Figure 생성 ──────────────────────────────────────
def make_weight_heatmap(weights: np.ndarray) -> "plt.Figure":
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(
        weights, annot=True, fmt='.3f', cmap='YlOrRd',
        xticklabels=SENT_FEATS, yticklabels=PURPOSES,
        ax=ax, vmin=0, vmax=0.6, annot_kws={'size': 11},
    )
    ax.set_title('Learned Purpose Weights', fontweight='bold')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Purpose')
    plt.tight_layout()
    return fig


# ── 메인 학습 함수 ───────────────────────────────────────────────
def train(
    cfg        : dict,
    train_ds,
    test_ds,
    device     : torch.device,
) -> Tuple[DualHeadModelV2, Dict]:
    """
    전체 학습 루프.

    Args:
        cfg      : config.yaml 딕셔너리
        train_ds : ProficiencyDataset (train)
        test_ds  : ProficiencyDataset (test)
        device   : cuda or cpu

    Returns:
        model        : 최적 모델
        final_metrics: 최종 평가 결과
    """
    # WandB 초기화
    use_wandb = cfg['wandb']['enabled']
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project = cfg['project']['name'],
                entity  = cfg['project']['entity'],
                name    = cfg['project']['run_name'],
                tags    = cfg['project']['tags'],
                config  = {
                    **cfg['model'],
                    **cfg['training'],
                    'dataset': cfg['data']['dataset'],
                },
            )
            print("[WandB] Run initialized.")
        except Exception as e:
            print(f"[WandB] Failed to initialize: {e}. Proceeding without WandB.")
            use_wandb = False

    t = cfg['training']
    epochs     = t['epochs']
    batch_size = t['batch_size']
    lr         = t['learning_rate']
    wd         = t['weight_decay']
    use_uw     = t['use_uncertainty_weighting']
    goal_w     = t['goal_loss_weight']
    cefr_w     = t['cefr_loss_weight']

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    model     = DualHeadModelV2(cfg).to(device)
    uw        = UncertaintyWeighting(num_tasks=2).to(device)

    # Uncertainty Weighting 파라미터도 함께 학습
    all_params = list(model.parameters()) + list(uw.parameters())
    optimizer  = optim.AdamW(all_params, lr=lr, weight_decay=wd)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    goal_crit = nn.MSELoss()
    cefr_crit = nn.CrossEntropyLoss()

    best_pcc   = -1.0
    best_state = None
    history    = []

    print(f"\n[Trainer] Starting training ({epochs} epochs, device={device})")
    print(f"  Uncertainty Weighting: {'ON' if use_uw else 'OFF'}")
    print(f"  WandB logging        : {'ON' if use_wandb else 'OFF'}")
    print()

    for ep in range(1, epochs + 1):
        train_logs = train_one_epoch(
            model, train_loader, optimizer,
            goal_crit, cefr_crit, uw, device,
            use_uw=use_uw, goal_w=goal_w, cefr_w=cefr_w,
        )
        test_logs  = evaluate(model, test_loader, device)
        scheduler.step()

        # 학습된 가중치
        weights_arr = get_learned_weights(model, device)

        log_dict = {
            **train_logs,
            **test_logs,
            'epoch': ep,
            'lr'   : scheduler.get_last_lr()[0],
        }
        history.append(log_dict)

        # WandB 로깅
        if use_wandb:
            import wandb
            wandb_log = dict(log_dict)
            if cfg['wandb']['log_weights']:
                fig = make_weight_heatmap(weights_arr)
                wandb_log['purpose_weights'] = wandb.Image(fig)
                plt.close(fig)
            wandb.log(wandb_log)

        # Best 모델 저장
        cur_pcc = test_logs['test/goal_pcc']
        if cur_pcc > best_pcc:
            best_pcc   = cur_pcc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # 콘솔 출력
        if ep % 10 == 0 or ep == 1:
            print(
                f"  Ep {ep:>3}/{epochs} | "
                f"Loss={train_logs['train/total_loss']:.4f} | "
                f"Goal PCC={cur_pcc:.4f} "
                f"RMSE={test_logs['test/goal_rmse']:.4f} | "
                f"CEFR Acc={test_logs['test/cefr_acc']:.4f} "
                f"(±1={test_logs['test/cefr_pm1']:.4f}) | "
                f"LW=[{train_logs['train/goal_weight']:.2f},"
                f"{train_logs['train/cefr_weight']:.2f}]"
            )

    # Best 모델 복원
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, test_loader, device)

    print(f"\n[Trainer] Best Results:")
    print(f"  Goal  — PCC:{final_metrics['test/goal_pcc']:.4f}  "
          f"RMSE:{final_metrics['test/goal_rmse']:.4f}  "
          f"MAE:{final_metrics['test/goal_mae']:.4f}")
    print(f"  CEFR  — Acc:{final_metrics['test/cefr_acc']:.4f}  "
          f"±1:{final_metrics['test/cefr_pm1']:.4f}")

    if use_wandb:
        import wandb
        wandb.summary.update({k: v for k, v in final_metrics.items()})
        wandb.finish()

    return model, history, final_metrics


# ── 결과 저장 ────────────────────────────────────────────────────
def save_results(
    model        : DualHeadModelV2,
    history      : List[Dict],
    final_metrics: Dict,
    scaler,
    cfg          : dict,
) -> None:
    """모델 체크포인트 + 학습 곡선 + 요약 JSON 저장."""
    save_dir = Path(cfg['output']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 모델 저장
    if cfg['output']['save_model']:
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_metrics'   : final_metrics,
            'scaler_mean'     : scaler.mean_.tolist(),
            'scaler_scale'    : scaler.scale_.tolist(),
            'config'          : cfg,
        }, save_dir / cfg['output']['model_filename'])
        print(f"[Save] Model: {save_dir / cfg['output']['model_filename']}")

    # 학습 곡선 저장
    _plot_history(history, save_dir)

    # 요약 JSON
    summary = {
        'final_metrics': final_metrics,
        'config'       : cfg['training'],
        'model'        : cfg['model'],
    }
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Save] Summary: {save_dir / 'summary.json'}")


def _plot_history(history: List[Dict], save_dir: Path) -> None:
    """학습 곡선 저장."""
    import pandas as pd
    hist = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Training History', fontsize=13, fontweight='bold')

    axes[0].plot(hist['epoch'], hist['train/total_loss'],
                 color='#4C72B0', lw=2)
    axes[0].set_title('Total Loss'); axes[0].set_xlabel('Epoch')
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist['epoch'], hist['test/goal_pcc'],
                 color='#55A868', lw=2, label='Goal PCC')
    axes[1].axhline(0.9655, color='red', ls='--', lw=1.2,
                    label='Ridge Baseline')
    axes[1].set_title('Goal PCC'); axes[1].set_xlabel('Epoch')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    axes[2].plot(hist['epoch'], hist['test/cefr_acc'],
                 color='#C44E52', lw=2, label='Exact')
    axes[2].plot(hist['epoch'], hist['test/cefr_pm1'],
                 color='#8172B2', lw=2, ls='--', label='±1')
    axes[2].set_title('CEFR Accuracy'); axes[2].set_xlabel('Epoch')
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = save_dir / 'training_history.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Save] History plot: {out}")
