"""
train.py
========
학습 실행 진입점.

사용법:
  # 기본 실행
  python src/train.py

  # config 파일 지정
  python src/train.py --config src/config.yaml

  # WandB 끄고 빠르게 테스트
  python src/train.py --no-wandb --epochs 5
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# 같은 src/ 폴더 내 모듈 import
sys.path.insert(0, str(Path(__file__).parent))
from dataset import ProficiencyDataset, load_speechocean
from trainer import save_results, train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description='Train Purpose-Conditioned Dual-Head Proficiency Model'
    )
    parser.add_argument('--config',    type=str,
                        default='src/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--no-wandb',  action='store_true',
                        help='Disable WandB logging')
    parser.add_argument('--epochs',    type=int, default=None,
                        help='Override epochs in config')
    parser.add_argument('--run-name',  type=str, default=None,
                        help='Override WandB run name')
    args = parser.parse_args()

    # ── config 로드 ──────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        # 프로젝트 루트에서 실행할 때 경로 조정
        config_path = Path(__file__).parent / 'config.yaml'

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # CLI 인자로 override
    if args.no_wandb:
        cfg['wandb']['enabled'] = False
    if args.epochs:
        cfg['training']['epochs'] = args.epochs
    if args.run_name:
        cfg['project']['run_name'] = args.run_name

    # ── 환경 설정 ────────────────────────────────────────────────
    set_seed(cfg['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 62)
    print("Goal-Aware Proficiency Model — Training")
    print("=" * 62)
    print(f"Config : {config_path}")
    print(f"Device : {device}")
    print(f"Epochs : {cfg['training']['epochs']}")
    print(f"WandB  : {'ON' if cfg['wandb']['enabled'] else 'OFF'}")
    print(f"UW     : {'ON' if cfg['training']['use_uncertainty_weighting'] else 'OFF'}")

    # ── 데이터 로드 ──────────────────────────────────────────────
    train_df, test_df = load_speechocean(cfg, verbose=True)

    train_ds = ProficiencyDataset(
        train_df,
        feature_cols = cfg['data']['feature_cols'],
        fit_scaler   = True,
    )
    test_ds = ProficiencyDataset(
        test_df,
        feature_cols = cfg['data']['feature_cols'],
        scaler       = train_ds.scaler,
    )

    # ── 학습 ─────────────────────────────────────────────────────
    model, history, final_metrics = train(cfg, train_ds, test_ds, device)

    # ── 저장 ─────────────────────────────────────────────────────
    save_results(model, history, final_metrics, train_ds.scaler, cfg)

    # ── 최종 출력 ────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("FINAL RESULTS")
    print("=" * 62)
    print(f"\n  Goal Score Prediction:")
    print(f"    PCC  : {final_metrics['test/goal_pcc']:.4f}")
    print(f"    RMSE : {final_metrics['test/goal_rmse']:.4f}")
    print(f"    MAE  : {final_metrics['test/goal_mae']:.4f}")
    print(f"\n  CEFR Level Classification:")
    print(f"    Exact Accuracy : {final_metrics['test/cefr_acc']:.4f}")
    print(f"    ±1   Accuracy  : {final_metrics['test/cefr_pm1']:.4f}")
    print(f"\n  Results saved to: {cfg['output']['save_dir']}/")


if __name__ == '__main__':
    main()
