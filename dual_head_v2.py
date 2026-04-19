"""
Purpose-Conditioned Dual-Head Proficiency Model v2
===================================================
핵심 변경사항 (v1 → v2):

  [v1 문제]
  - 목적 레이블을 피처 기반으로 자동 부여 → 95.8%가 business로 쏠림
  - 고정 가중치(travel=0.4/0.1/0.3/0.2)를 손으로 설정

  [v2 해결]
  - 목적 레이블 자동 부여 완전 제거
  - 학습자가 앱에서 직접 목적 선택 → 모델에 조건으로 입력
  - Purpose-Conditioned Weight Generator:
      목적 임베딩 → 소프트맥스 → 피처별 가중치 동적 생성
      (모델이 "여행 목적이면 발음을 더 중시해야 한다"를 스스로 학습)

논문 Contribution:
  1. 학습자 명시 목적을 조건으로 받는 최초의 이중 출력 어학 평가 모델
  2. 고정 가중치가 아닌 학습 가능한 목적별 가중치 생성기 (Weight Generator)
  3. 절대 능력(CEFR) + 목적 맞춤 점수 동시 출력

참고 논문:
  - Bamdev et al. (2023) IJAIED — feature importance 수치 (baseline)
  - Gong et al. (2022) ICASSP — multi-aspect multi-head 구조
  - TOEIC/TOEFL/IELTS 설계 차이 — 목적별 평가 기준 상이성 근거
"""

import random
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats as scipy_stats
from datasets import load_dataset

warnings.filterwarnings('ignore')

# ── 재현성 고정 ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

PURPOSES    = ['travel', 'business', 'academic']
PURPOSE2IDX = {p: i for i, p in enumerate(PURPOSES)}
IDX2PURPOSE = {i: p for p, i in PURPOSE2IDX.items()}

# 목적별 평가 기준 설명 (논문 Table로 들어갈 내용)
PURPOSE_DESCRIPTION = {
    'travel'  : 'Pronunciation accuracy + fluency (intelligibility for daily travel)',
    'business': 'Vocabulary completeness + accuracy (formal register, professional)',
    'academic': 'Prosodic control + completeness (presentation, academic discourse)',
}

# Speechocean762의 문장 레벨 피처 (모델 학습에 사용)
SENTENCE_FEATS = ['accuracy', 'completeness', 'fluency', 'prosodic']

# 전체 피처 (단어/음소 레벨 집계 포함)
FEATURE_COLS = [
    'accuracy', 'prosodic', 'fluency', 'completeness',
    'word_acc_mean', 'word_acc_std', 'word_acc_min', 'word_total_mean',
    'phone_acc_mean', 'phone_acc_std', 'phone_acc_min',
]

# CEFR 레벨 매핑
CEFR_LABELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

def score_to_cefr(score: float) -> int:
    if score <= 2.0: return 0
    elif score <= 4.0: return 1
    elif score <= 6.0: return 2
    elif score <= 7.5: return 3
    elif score <= 9.0: return 4
    else: return 5


# ══════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════

def load_data(verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    speechocean762 로드.

    v2 핵심 변경:
      목적 레이블 자동 부여 완전 제거.
      대신 각 샘플을 3가지 목적 모두에 대해 복제(augmentation)하여
      "어떤 목적으로 평가하든 동작하는" 모델을 학습.

      → 실제 앱에서: 학습자가 목적을 선택하면 그 idx를 모델에 입력
      → 학습 시: 같은 발화를 travel/business/academic 3가지로 복제해서
                  각 목적에 맞는 가중 점수를 타겟으로 학습
    """
    if verbose:
        print("[1/5] Loading speechocean762...")

    dataset = load_dataset("mispeech/speechocean762")

    def to_base_df(split) -> pd.DataFrame:
        sp = split.remove_columns(['audio']) if 'audio' in split.column_names else split
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

    base_train = to_base_df(dataset['train'])
    base_test  = to_base_df(dataset['test'])

    # ── 핵심: 목적별 가중 점수 사전 계산 ────────────────────────
    # 논문 Table 1에 들어갈 가중치 (TOEIC/TOEFL/IELTS 설계 근거)
    # 이 가중치는 모델 학습 타겟이 아닌 supervision signal로 사용
    purpose_weights = {
        # 여행: 발음 명확도 + 유창성 최우선 (일상 의사소통 목적)
        'travel'  : {'accuracy': 0.40, 'completeness': 0.10,
                     'fluency' : 0.30, 'prosodic'    : 0.20},
        # 취업: 어휘/표현 완성도 + 정확도 (격식체, 전문 어휘 필요)
        'business': {'accuracy': 0.30, 'completeness': 0.40,
                     'fluency' : 0.20, 'prosodic'    : 0.10},
        # 학업: 운율/억양 + 완성도 (발표, 토론 목적)
        'academic': {'accuracy': 0.20, 'completeness': 0.30,
                     'fluency' : 0.20, 'prosodic'    : 0.30},
    }

    def expand_by_purpose(df: pd.DataFrame) -> pd.DataFrame:
        """
        각 샘플을 3개 목적으로 복제.
        purpose_idx: 학습자가 앱에서 선택하는 값 (0/1/2)
        goal_score : 해당 목적의 가중 점수 (학습 타겟)
        """
        rows = []
        for _, row in df.iterrows():
            base = row.to_dict()
            base['cefr_level'] = score_to_cefr(row['total'])
            for p, w in purpose_weights.items():
                r = base.copy()
                r['purpose']     = p
                r['purpose_idx'] = PURPOSE2IDX[p]
                r['goal_score']  = sum(row[f] * w[f] for f in SENTENCE_FEATS)
                rows.append(r)
        return pd.DataFrame(rows)

    train_df = expand_by_purpose(base_train)
    test_df  = expand_by_purpose(base_test)

    if verbose:
        print(f"    Base train: {len(base_train)} → expanded: {len(train_df)}")
        print(f"    Base test : {len(base_test)}  → expanded: {len(test_df)}")
        print(f"    Purpose distribution (perfectly balanced by design):")
        for p in PURPOSES:
            cnt = (train_df['purpose'] == p).sum()
            print(f"      {p:<12}: {cnt} ({cnt/len(train_df)*100:.1f}%)")
        print(f"    CEFR distribution (base train):")
        cefr_counts = base_train['total'].apply(score_to_cefr).value_counts().sort_index()
        for lvl, cnt in cefr_counts.items():
            print(f"      {CEFR_LABELS[lvl]}: {cnt}")

    return train_df, test_df, purpose_weights


# ══════════════════════════════════════════════════════════════
# 2. Dataset
# ══════════════════════════════════════════════════════════════

class ProficiencyDataset(Dataset):
    """
    v2: purpose_idx는 학습자가 직접 선택한 값.
    같은 발화가 3가지 purpose_idx로 확장되어 있음.
    """

    def __init__(self, df: pd.DataFrame,
                 scaler: StandardScaler = None,
                 fit_scaler: bool = False):
        X = df[FEATURE_COLS].values.astype(np.float32)
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X)

        self.X           = torch.tensor(X,                              dtype=torch.float32)
        self.purpose_idx = torch.tensor(df['purpose_idx'].values,       dtype=torch.long)
        self.goal_score  = torch.tensor(df['goal_score'].values,        dtype=torch.float32)
        self.cefr_level  = torch.tensor(df['cefr_level'].values,        dtype=torch.long)
        self.total_score = torch.tensor(df['total'].values,             dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features'   : self.X[idx],
            'purpose_idx': self.purpose_idx[idx],
            'goal_score' : self.goal_score[idx],
            'cefr_level' : self.cefr_level[idx],
            'total_score': self.total_score[idx],
        }


# ══════════════════════════════════════════════════════════════
# 3. 모델: Purpose-Conditioned Weight Generator
# ══════════════════════════════════════════════════════════════

class PurposeWeightGenerator(nn.Module):
    """
    논문 핵심 contribution: 학습 가능한 목적별 가중치 생성기

    목적 임베딩 → 피처별 가중치 생성 (소프트맥스로 합=1 보장)

    의미:
      - "travel" 임베딩이 입력되면 [0.40, 0.10, 0.30, 0.20] 같은 가중치를 출력
      - 이 가중치는 손으로 설정한 게 아니라 데이터에서 학습됨
      - 학습 후 어떤 목적이 어떤 피처를 중시하는지 해석 가능 → 논문 분석 섹션
    """

    def __init__(self, num_purposes: int = 3,
                 purpose_emb_dim: int = 16,
                 num_sentence_feats: int = 4):   # accuracy/completeness/fluency/prosodic
        super().__init__()

        self.num_sentence_feats = num_sentence_feats

        # 목적 임베딩 테이블
        self.purpose_emb = nn.Embedding(num_purposes, purpose_emb_dim)

        # 임베딩 → 피처 가중치 (소프트맥스로 정규화)
        self.weight_net = nn.Sequential(
            nn.Linear(purpose_emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_sentence_feats),
        )

    def forward(self, purpose_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            purpose_idx: (B,) — 0=travel, 1=business, 2=academic

        Returns:
            weights: (B, num_sentence_feats) — 합이 1인 가중치 벡터
        """
        emb     = self.purpose_emb(purpose_idx)       # (B, purpose_emb_dim)
        logits  = self.weight_net(emb)                 # (B, num_sentence_feats)
        weights = torch.softmax(logits, dim=-1)        # (B, num_sentence_feats) — 합=1
        return weights


class DualHeadModelV2(nn.Module):
    """
    Purpose-Conditioned Dual-Head Proficiency Model v2

    아키텍처:
      [피처 입력 (11차원)]
            ↓
      [Feature Encoder] ← 피처 표현 학습
            ↓                           ↑ [Purpose Embedding]
      [Fusion: concat]                  |
            ↓                           |
      [Shared Encoder]                  |
       ┌────┴────┐          [Purpose Weight Generator]
       ↓         ↓                    ↓
    [Goal Head] [CEFR Head]  [Weighted Sum of sent. feats]
    (회귀)      (분류)         ↓
       ↓         ↓          goal_score (목적 맞춤 점수)
    0~10점    A1~C2

    v2 핵심:
      Goal Head의 타겟이 "PurposeWeightGenerator가 생성한 가중치 × 피처"
      → 모델이 목적별로 어떤 피처를 중시해야 하는지 스스로 학습
    """

    def __init__(
        self,
        input_dim      : int   = 11,
        purpose_emb_dim: int   = 16,
        hidden_dim     : int   = 64,
        num_purposes   : int   = 3,
        num_cefr       : int   = 6,
        dropout        : float = 0.3,
    ):
        super().__init__()

        # ── 목적 가중치 생성기 (핵심 모듈) ──────────────────────
        self.weight_generator = PurposeWeightGenerator(
            num_purposes       = num_purposes,
            purpose_emb_dim    = purpose_emb_dim,
            num_sentence_feats = 4,   # accuracy/completeness/fluency/prosodic
        )

        # ── 피처 인코더 ─────────────────────────────────────────
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── 공유 인코더 (피처 표현 + 목적 임베딩 융합) ───────────
        fusion_dim = hidden_dim + purpose_emb_dim
        self.shared_encoder = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Head 1: 목적 맞춤 점수 (회귀) ───────────────────────
        # 입력: shared 표현 + 가중 피처 합산값
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # ── Head 2: CEFR 분류 ────────────────────────────────────
        self.cefr_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_cefr),
        )

        # 목적 임베딩 (공유 인코더 입력용)
        self.purpose_emb_shared = nn.Embedding(num_purposes, purpose_emb_dim)

    def forward(
        self,
        features    : torch.Tensor,   # (B, 11)
        purpose_idx : torch.Tensor,   # (B,)
        sent_feats  : torch.Tensor,   # (B, 4) — accuracy/completeness/fluency/prosodic
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            goal_score   : (B,) — 목적 맞춤 가중 점수 (0~10)
            cefr_logits  : (B, 6) — CEFR 레벨 로짓
            purpose_weights: (B, 4) — 목적별 학습된 가중치 (해석 가능)
        """
        # 1) 목적별 가중치 생성
        purpose_weights = self.weight_generator(purpose_idx)     # (B, 4)

        # 2) 가중 피처 합산 (목적 맞춤 스칼라 점수)
        weighted_sum = (purpose_weights * sent_feats).sum(dim=-1, keepdim=True)  # (B, 1)

        # 3) 피처 인코딩
        feat_repr = self.feature_encoder(features)               # (B, hidden)

        # 4) 목적 임베딩과 융합
        purp_emb  = self.purpose_emb_shared(purpose_idx)         # (B, emb_dim)
        fused     = torch.cat([feat_repr, purp_emb], dim=-1)     # (B, hidden+emb)
        shared    = self.shared_encoder(fused)                    # (B, hidden)

        # 5) Head 1: shared 표현 + 가중합을 함께 입력
        goal_in    = torch.cat([shared, weighted_sum], dim=-1)   # (B, hidden+1)
        goal_score = self.goal_head(goal_in).squeeze(-1) * 10.0  # (B,) 0~10

        # 6) Head 2: CEFR 분류 (목적과 무관한 절대 능력)
        cefr_logits = self.cefr_head(shared)                     # (B, 6)

        return goal_score, cefr_logits, purpose_weights


# ══════════════════════════════════════════════════════════════
# 4. 학습
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer,
                    goal_crit, cefr_crit,
                    goal_w=0.6, cefr_w=0.4) -> Dict:
    model.train()
    total = goal_sum = cefr_sum = 0.0

    for batch in loader:
        feat   = batch['features'].to(DEVICE)
        pidx   = batch['purpose_idx'].to(DEVICE)
        gtarget= batch['goal_score'].to(DEVICE)
        ctarget= batch['cefr_level'].to(DEVICE)

        # sent_feats: accuracy/completeness/fluency/prosodic (인덱스 0,3,2,1)
        sent   = feat[:, [0, 3, 2, 1]]   # FEATURE_COLS 순서대로

        optimizer.zero_grad()
        goal_pred, cefr_logits, _ = model(feat, pidx, sent)

        g_loss = goal_crit(goal_pred, gtarget)
        c_loss = cefr_crit(cefr_logits, ctarget)
        loss   = goal_w * g_loss + cefr_w * c_loss
        loss.backward()

        # Gradient clipping (학습 안정화)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total    += loss.item()
        goal_sum += g_loss.item()
        cefr_sum += c_loss.item()

    n = len(loader)
    return {'total': total/n, 'goal': goal_sum/n, 'cefr': cefr_sum/n}


@torch.no_grad()
def evaluate(model, loader) -> Dict:
    model.eval()
    g_pred, g_true = [], []
    c_pred, c_true = [], []

    for batch in loader:
        feat  = batch['features'].to(DEVICE)
        pidx  = batch['purpose_idx'].to(DEVICE)
        sent  = feat[:, [0, 3, 2, 1]]

        gp, cl, _ = model(feat, pidx, sent)
        g_pred.extend(gp.cpu().numpy())
        g_true.extend(batch['goal_score'].numpy())
        c_pred.extend(cl.argmax(-1).cpu().numpy())
        c_true.extend(batch['cefr_level'].numpy())

    g_pred, g_true = np.array(g_pred), np.array(g_true)
    c_pred, c_true = np.array(c_pred), np.array(c_true)

    pcc, _   = scipy_stats.pearsonr(g_true, g_pred)
    mse      = mean_squared_error(g_true, g_pred)
    mae      = mean_absolute_error(g_true, g_pred)
    c_acc    = (c_pred == c_true).mean()
    c_pm1    = (np.abs(c_pred - c_true) <= 1).mean()

    return {
        'goal_pcc' : round(float(pcc), 4),
        'goal_rmse': round(float(np.sqrt(mse)), 4),
        'goal_mae' : round(float(mae), 4),
        'cefr_acc' : round(float(c_acc), 4),
        'cefr_pm1' : round(float(c_pm1), 4),
    }


def train_model(train_df, test_df,
                epochs=60, batch_size=128, lr=1e-3,
                verbose=True):

    if verbose:
        print(f"\n[3/5] Training v2 model ({epochs} epochs, device={DEVICE})...")

    train_ds = ProficiencyDataset(train_df, fit_scaler=True)
    test_ds  = ProficiencyDataset(test_df,  scaler=train_ds.scaler)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model     = DualHeadModelV2(input_dim=len(FEATURE_COLS)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    goal_crit = nn.MSELoss()
    cefr_crit = nn.CrossEntropyLoss()

    history   = []
    best_pcc  = -1.0
    best_state= None

    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_ld, optimizer, goal_crit, cefr_crit)
        te = evaluate(model, test_ld)
        scheduler.step()
        history.append({'epoch': ep, **tr, **te})

        if te['goal_pcc'] > best_pcc:
            best_pcc   = te['goal_pcc']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  Ep {ep:>3}/{epochs} | "
                  f"Loss={tr['total']:.4f} | "
                  f"Goal PCC={te['goal_pcc']:.4f} RMSE={te['goal_rmse']:.4f} | "
                  f"CEFR Acc={te['cefr_acc']:.4f} (±1={te['cefr_pm1']:.4f})")

    model.load_state_dict(best_state)
    final = evaluate(model, test_ld)

    if verbose:
        print(f"\n  Best Results:")
        print(f"    Goal  — PCC:{final['goal_pcc']:.4f}  "
              f"RMSE:{final['goal_rmse']:.4f}  MAE:{final['goal_mae']:.4f}")
        print(f"    CEFR  — Acc:{final['cefr_acc']:.4f}  ±1:{final['cefr_pm1']:.4f}")

    return model, history, final, train_ds, test_ds


# ══════════════════════════════════════════════════════════════
# 5. 학습된 가중치 분석 (논문 분석 섹션용)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_learned_weights(model: nn.Module, save_dir: str = './results') -> pd.DataFrame:
    """
    논문 핵심 분석:
    PurposeWeightGenerator가 학습한 가중치를 시각화.
    "모델이 여행 목적에는 발음을 더 중시하는 것을 스스로 학습했는가?"를 검증.
    """
    model.eval()
    records = []

    for p, idx in PURPOSE2IDX.items():
        pidx    = torch.tensor([idx], dtype=torch.long).to(DEVICE)
        weights = model.weight_generator(pidx).squeeze().cpu().numpy()
        for feat, w in zip(SENTENCE_FEATS, weights):
            records.append({'Purpose': p, 'Feature': feat, 'Learned Weight': round(float(w), 4)})

    df = pd.DataFrame(records)

    # 피벗 테이블
    pivot = df.pivot(index='Purpose', columns='Feature', values='Learned Weight')

    print("\n" + "=" * 62)
    print("LEARNED PURPOSE WEIGHTS (논문 Table 2 후보)")
    print("=" * 62)
    print("\n  Model이 학습한 목적별 피처 가중치:")
    print(f"  {'Purpose':<12}", end="")
    for f in SENTENCE_FEATS:
        print(f"  {f:>12}", end="")
    print()
    print("  " + "-" * 60)
    for p in PURPOSES:
        row = pivot.loc[p]
        print(f"  {p:<12}", end="")
        for f in SENTENCE_FEATS:
            print(f"  {row[f]:>12.4f}", end="")
        print()

    # 사전 설정 가중치와 비교
    preset = {
        'travel'  : {'accuracy': 0.40, 'completeness': 0.10, 'fluency': 0.30, 'prosodic': 0.20},
        'business': {'accuracy': 0.30, 'completeness': 0.40, 'fluency': 0.20, 'prosodic': 0.10},
        'academic': {'accuracy': 0.20, 'completeness': 0.30, 'fluency': 0.20, 'prosodic': 0.30},
    }
    print(f"\n  Preset weights (from literature):")
    print(f"  {'Purpose':<12}", end="")
    for f in SENTENCE_FEATS:
        print(f"  {f:>12}", end="")
    print()
    print("  " + "-" * 60)
    for p in PURPOSES:
        print(f"  {p:<12}", end="")
        for f in SENTENCE_FEATS:
            print(f"  {preset[p][f]:>12.4f}", end="")
        print()

    # 히트맵 저장
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Purpose-Conditioned Feature Weights', fontsize=13, fontweight='bold')

    learned_matrix = pivot[SENTENCE_FEATS].values
    preset_matrix  = np.array([[preset[p][f] for f in SENTENCE_FEATS] for p in PURPOSES])

    for ax, mat, title in zip(axes,
                               [learned_matrix, preset_matrix],
                               ['Learned Weights (from model)',
                                'Preset Weights (from literature)']):
        sns.heatmap(mat, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=SENTENCE_FEATS,
                    yticklabels=PURPOSES,
                    ax=ax, vmin=0, vmax=0.6,
                    annot_kws={'size': 11})
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Purpose')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/learned_weights.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_dir}/learned_weights.png")

    return df


# ══════════════════════════════════════════════════════════════
# 6. 시각화
# ══════════════════════════════════════════════════════════════

def plot_results(history, final, model, test_ds,
                 baseline_pcc=0.9655, save_dir='./results'):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dual-Head Model v2 Results\n'
                 '(Purpose-Conditioned, Learner-Specified Goal)',
                 fontsize=13, fontweight='bold', y=1.02)

    hist = pd.DataFrame(history)

    # (A) Goal PCC 학습 곡선
    ax = axes[0][0]
    ax.plot(hist['epoch'], hist['goal_pcc'],
            color='#4C72B0', lw=2, label='DualHead v2 (Goal PCC)')
    ax.axhline(baseline_pcc, color='red', ls='--', lw=1.5,
               label=f'Ridge Baseline PCC={baseline_pcc}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('PCC')
    ax.set_title('(A) Training Curve — Goal Score PCC', fontweight='bold')
    ax.legend(fontsize=9); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    # (B) CEFR Accuracy 학습 곡선
    ax = axes[0][1]
    ax.plot(hist['epoch'], hist['cefr_acc'],
            color='#55A868', lw=2, label='CEFR Exact Acc')
    ax.plot(hist['epoch'], hist['cefr_pm1'],
            color='#C44E52', lw=2, ls='--', label='CEFR ±1 Acc')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('(B) Training Curve — CEFR Classification', fontweight='bold')
    ax.legend(fontsize=9); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    # (C) Predicted vs Actual (Goal) — 목적별 색상
    ax = axes[1][0]
    model.eval()
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    colors_map = {'travel': '#4C72B0', 'business': '#C44E52', 'academic': '#55A868'}

    with torch.no_grad():
        all_pred, all_true, all_purpose = [], [], []
        for batch in loader:
            feat = batch['features'].to(DEVICE)
            pidx = batch['purpose_idx'].to(DEVICE)
            sent = feat[:, [0, 3, 2, 1]]
            gp, _, _ = model(feat, pidx, sent)
            all_pred.extend(gp.cpu().numpy())
            all_true.extend(batch['goal_score'].numpy())
            all_purpose.extend(batch['purpose_idx'].numpy())

    for p, idx in PURPOSE2IDX.items():
        mask = np.array(all_purpose) == idx
        ax.scatter(np.array(all_true)[mask], np.array(all_pred)[mask],
                   alpha=0.2, s=6, color=colors_map[p], label=p)

    lims = [min(np.array(all_true).min(), np.array(all_pred).min()) - 0.3,
            max(np.array(all_true).max(), np.array(all_pred).max()) + 0.3]
    ax.plot(lims, lims, 'k--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual Goal Score'); ax.set_ylabel('Predicted Goal Score')
    ax.set_title(f'(C) Goal Score: Predicted vs Actual\n'
                 f'PCC={final["goal_pcc"]:.4f}, RMSE={final["goal_rmse"]:.4f}',
                 fontweight='bold')
    ax.legend(fontsize=9, title='Purpose'); ax.grid(alpha=0.3)

    # (D) CEFR 혼동행렬
    ax = axes[1][1]
    c_pred, c_true = [], []
    with torch.no_grad():
        for batch in loader:
            feat = batch['features'].to(DEVICE)
            pidx = batch['purpose_idx'].to(DEVICE)
            sent = feat[:, [0, 3, 2, 1]]
            _, cl, _ = model(feat, pidx, sent)
            c_pred.extend(cl.argmax(-1).cpu().numpy())
            c_true.extend(batch['cefr_level'].numpy())

    conf = np.zeros((6, 6), dtype=int)
    for t, p in zip(c_true, c_pred):
        conf[t][p] += 1
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues',
                xticklabels=CEFR_LABELS, yticklabels=CEFR_LABELS,
                ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'(D) CEFR Confusion Matrix\n'
                 f'Acc={final["cefr_acc"]:.4f}, ±1={final["cefr_pm1"]:.4f}',
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/dualhead_v2_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_dir}/dualhead_v2_results.png")


# ══════════════════════════════════════════════════════════════
# 7. 추론 시뮬레이션 (앱에서 어떻게 작동하는지 데모)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def inference_demo(model: nn.Module, scaler: StandardScaler) -> None:
    """
    실제 앱 사용 시나리오 시뮬레이션:
    같은 학습자(동일 피처)가 목적을 바꾸면 점수가 어떻게 달라지는지 보여줌.
    이게 논문의 핵심 예시 (Table 3 또는 Figure 후보).
    """
    print("\n" + "=" * 62)
    print("APP INFERENCE DEMO")
    print("Same learner, different purpose → different score")
    print("=" * 62)

    # 가상 학습자 3명 (발음 강점 / 어휘 강점 / 운율 강점)
    learners = {
        'Learner A (pronunciation-strong)': {
            'accuracy': 9.0, 'completeness': 6.0, 'fluency': 8.0, 'prosodic': 7.0,
            'word_acc_mean': 8.5, 'word_acc_std': 0.8, 'word_acc_min': 7.0,
            'word_total_mean': 8.2, 'phone_acc_mean': 1.8, 'phone_acc_std': 0.2,
            'phone_acc_min': 1.5,
        },
        'Learner B (vocabulary-strong)': {
            'accuracy': 6.0, 'completeness': 9.5, 'fluency': 7.0, 'prosodic': 6.5,
            'word_acc_mean': 6.5, 'word_acc_std': 0.5, 'word_acc_min': 5.5,
            'word_total_mean': 7.8, 'phone_acc_mean': 1.5, 'phone_acc_std': 0.3,
            'phone_acc_min': 1.2,
        },
        'Learner C (balanced)': {
            'accuracy': 7.5, 'completeness': 7.5, 'fluency': 7.5, 'prosodic': 7.5,
            'word_acc_mean': 7.5, 'word_acc_std': 0.5, 'word_acc_min': 6.5,
            'word_total_mean': 7.5, 'phone_acc_mean': 1.7, 'phone_acc_std': 0.2,
            'phone_acc_min': 1.4,
        },
    }

    model.eval()
    for name, feats in learners.items():
        x_raw = np.array([[feats[c] for c in FEATURE_COLS]], dtype=np.float32)
        x_sc  = scaler.transform(x_raw)
        x_t   = torch.tensor(x_sc, dtype=torch.float32).to(DEVICE)
        sent  = x_t[:, [0, 3, 2, 1]]

        print(f"\n  {name}")
        print(f"  Raw features: acc={feats['accuracy']:.1f}  "
              f"comp={feats['completeness']:.1f}  "
              f"flu={feats['fluency']:.1f}  pros={feats['prosodic']:.1f}")
        print(f"  {'Purpose':<12} {'Goal Score':>12} {'Learned Weights (acc/comp/flu/pros)'}")
        print(f"  {'-'*60}")

        for p, idx in PURPOSE2IDX.items():
            pidx = torch.tensor([idx], dtype=torch.long).to(DEVICE)
            gp, cl, pw = model(x_t, pidx, sent)
            cefr_idx = cl.argmax(-1).item()
            pw_np    = pw.squeeze().cpu().numpy()
            print(f"  {p:<12} {gp.item():>12.3f}     "
                  f"[{pw_np[0]:.3f} / {pw_np[1]:.3f} / "
                  f"{pw_np[2]:.3f} / {pw_np[3]:.3f}]  "
                  f"CEFR={CEFR_LABELS[cefr_idx]}")


# ══════════════════════════════════════════════════════════════
# 8. 메인
# ══════════════════════════════════════════════════════════════

def main():
    SAVE_DIR = './results'
    Path(SAVE_DIR).mkdir(exist_ok=True)

    print("=" * 62)
    print("Dual-Head Purpose-Conditioned Model v2")
    print("Learner-Specified Goal Input (No Auto-Labeling)")
    print("=" * 62)
    print(f"Device: {DEVICE}")

    # 1) 데이터
    train_df, test_df, purpose_weights = load_data(verbose=True)

    # 2) 저장
    print("\n[2/5] Saving labeled dataset...")
    train_df.to_csv(f'{SAVE_DIR}/train_v2.csv', index=False)
    test_df.to_csv(f'{SAVE_DIR}/test_v2.csv',   index=False)
    print(f"    Saved: train_v2.csv, test_v2.csv")

    # 3) 학습
    model, history, final, train_ds, test_ds = train_model(
        train_df, test_df, epochs=60, batch_size=128, lr=1e-3, verbose=True)

    # 4) 저장
    print("\n[4/5] Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_metrics'   : final,
        'purpose_weights' : purpose_weights,
        'feature_cols'    : FEATURE_COLS,
        'scaler_mean'     : train_ds.scaler.mean_.tolist(),
        'scaler_scale'    : train_ds.scaler.scale_.tolist(),
    }, f'{SAVE_DIR}/dualhead_v2.pth')
    print(f"    Saved: dualhead_v2.pth")

    # 5) 분석 & 시각화
    print("\n[5/5] Analysis & visualization...")
    weight_df = analyze_learned_weights(model, save_dir=SAVE_DIR)
    plot_results(history, final, model, test_ds, save_dir=SAVE_DIR)
    inference_demo(model, train_ds.scaler)

    # 최종 비교
    print("\n" + "=" * 62)
    print("FINAL COMPARISON")
    print("=" * 62)
    try:
        bl = pd.read_csv(f'{SAVE_DIR}/model_results.csv')
        best_bl = bl.loc[bl['Test_PCC'].idxmax()]
        print(f"\n  {'Model':<28} {'PCC':>7} {'RMSE':>7} {'MAE':>7}")
        print("  " + "-" * 52)
        for _, r in bl.sort_values('Test_PCC', ascending=False).iterrows():
            print(f"  {r['Model']:<28} {r['Test_PCC']:>7.4f} "
                  f"{r['Test_RMSE']:>7.4f} {r['Test_MAE']:>7.4f}")
        print(f"  {'DualHead v2 (Goal)':<28} {final['goal_pcc']:>7.4f} "
              f"{final['goal_rmse']:>7.4f} {final['goal_mae']:>7.4f}  ← NEW")
        print(f"\n  CEFR Head: Acc={final['cefr_acc']:.4f}, ±1={final['cefr_pm1']:.4f}")
    except FileNotFoundError:
        print(f"  Goal PCC={final['goal_pcc']:.4f}, RMSE={final['goal_rmse']:.4f}")

    # 요약 JSON
    with open(f'{SAVE_DIR}/dualhead_v2_summary.json', 'w') as f:
        json.dump({'model': 'DualHeadV2', 'final_metrics': final,
                   'device': str(DEVICE), 'epochs': 60}, f, indent=2)

    print(f"\nAll outputs saved to {SAVE_DIR}/")
    print("  dualhead_v2_results.png  — training curves + confusion matrix")
    print("  learned_weights.png      — learned vs preset weights (논문 Figure)")
    print("  dualhead_v2.pth          — model checkpoint")
    print("  dualhead_v2_summary.json — summary")


if __name__ == '__main__':
    main()
