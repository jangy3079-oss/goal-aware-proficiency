"""
model.py
========
모델 아키텍처 정의.

핵심 모듈:
  1. PurposeWeightGenerator  — 목적 임베딩 → 피처별 가중치 동적 생성
  2. DualHeadModelV2         — 공유 인코더 + Goal Head + CEFR Head
  3. UncertaintyWeighting    — 멀티태스크 손실 자동 조정 (Kendall et al., 2018)
"""

from typing import Tuple

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════
# 1. Purpose Weight Generator (핵심 contribution)
# ══════════════════════════════════════════════════════════════

class PurposeWeightGenerator(nn.Module):
    """
    학습 가능한 목적별 가중치 생성기.

    논문 contribution:
      고정 가중치(hand-crafted)가 아닌 데이터로부터 학습된 가중치를 사용.
      학습 후 히트맵으로 시각화 → "여행 목적이면 발음을 더 중시한다"는 것을
      모델이 데이터에서 스스로 발견했는지 검증 (논문 Figure 2 후보).

    구조:
      목적 인덱스 → 임베딩 → FC → Softmax → 피처별 가중치
      (소프트맥스로 가중치 합 = 1 보장)
    """

    def __init__(
        self,
        num_purposes      : int = 3,
        purpose_emb_dim   : int = 16,
        num_sentence_feats: int = 4,   # accuracy/completeness/fluency/prosodic
    ):
        super().__init__()
        self.purpose_emb = nn.Embedding(num_purposes, purpose_emb_dim)
        self.weight_net  = nn.Sequential(
            nn.Linear(purpose_emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_sentence_feats),
        )

    def forward(self, purpose_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            purpose_idx: (B,) — 0=travel, 1=business, 2=academic
        Returns:
            weights: (B, 4) — 합이 1인 피처 가중치
        """
        emb    = self.purpose_emb(purpose_idx)       # (B, emb_dim)
        logits = self.weight_net(emb)                 # (B, 4)
        return torch.softmax(logits, dim=-1)          # (B, 4)


# ══════════════════════════════════════════════════════════════
# 2. Uncertainty Weighting (Kendall et al., 2018)
# ══════════════════════════════════════════════════════════════

class UncertaintyWeighting(nn.Module):
    """
    멀티태스크 학습에서 각 태스크의 손실 가중치를 자동으로 조정.

    논문 근거:
      Kendall, A., Gal, Y., & Cipolla, R. (2018).
      "Multi-task learning using uncertainty to weigh losses
       for scene geometry and semantics." CVPR.

    수식:
      L = (1/2σ₁²) * L_goal + log(σ₁)
        + (1/2σ₂²) * L_cefr + log(σ₂)
      σ₁, σ₂: 학습 가능한 불확실성 파라미터

    의미:
      모델이 스스로 "지금은 goal 점수 학습이 더 중요하다"
      또는 "지금은 CEFR 분류가 더 중요하다"를 판단해서 가중치를 조정.
      이게 고정 가중치(0.6/0.4)보다 논문에서 더 가치 있는 contribution.
    """

    def __init__(self, num_tasks: int = 2):
        super().__init__()
        # log(σ²) 를 학습 (수치 안정성을 위해 σ 대신 log_var 사용)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        losses: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            losses: (goal_loss, cefr_loss)
        Returns:
            total_loss : 가중 합산 손실
            weights    : 각 태스크의 실제 가중치 (로깅용)
        """
        total = torch.tensor(0.0, device=self.log_vars.device)
        weights = []

        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])   # 1/σ²
            weighted  = precision * loss + self.log_vars[i]
            total    = total + weighted
            weights.append(float(precision.detach().cpu()))

        return total, weights


# ══════════════════════════════════════════════════════════════
# 3. Dual Head Model v2
# ══════════════════════════════════════════════════════════════

class DualHeadModelV2(nn.Module):
    """
    Purpose-Conditioned Dual-Head Proficiency Model v2.

    아키텍처:
      [피처 입력 (11D)] + [Purpose Embedding]
              ↓
        [Feature Encoder]
              ↓ (concat with purpose emb)
        [Shared Encoder]
         ┌────┴────┐
         ↓         ↓
      [Goal Head] [CEFR Head]
      목적 맞춤    절대 능력
       0~10점     A1~C2

    논문 contribution:
      1. 학습자 명시 목적을 조건으로 받는 이중 출력 어학 평가 모델
      2. Purpose Weight Generator: 고정 가중치 → 학습 가능한 가중치
      3. Uncertainty Weighting: 고정 loss 비율 → 자동 조정
    """

    def __init__(self, cfg: dict):
        super().__init__()

        m = cfg['model']
        input_dim       = m['input_dim']
        purpose_emb_dim = m['purpose_emb_dim']
        hidden_dim      = m['hidden_dim']
        num_purposes    = m['num_purposes']
        num_cefr        = m['num_cefr']
        dropout         = m['dropout']

        # ── 목적 가중치 생성기 ──────────────────────────────────
        self.weight_generator = PurposeWeightGenerator(
            num_purposes       = num_purposes,
            purpose_emb_dim    = purpose_emb_dim,
            num_sentence_feats = 4,
        )

        # ── 피처 인코더 ─────────────────────────────────────────
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── 공유 인코더 (피처 + 목적 임베딩 융합) ────────────────
        self.purpose_emb_shared = nn.Embedding(num_purposes, purpose_emb_dim)
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

    def forward(
        self,
        features    : torch.Tensor,   # (B, 11)
        purpose_idx : torch.Tensor,   # (B,)
        sent_feats  : torch.Tensor,   # (B, 4) accuracy/completeness/fluency/prosodic
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            goal_score    : (B,) — 0~10
            cefr_logits   : (B, 6)
            purpose_weights: (B, 4) — 학습된 가중치 (해석/시각화용)
        """
        # 1) 목적별 가중치 생성
        pw = self.weight_generator(purpose_idx)               # (B, 4)

        # 2) 가중 피처 합산
        weighted_sum = (pw * sent_feats).sum(dim=-1, keepdim=True)  # (B, 1)

        # 3) 피처 인코딩
        feat_repr = self.feature_encoder(features)            # (B, hidden)

        # 4) 목적 임베딩과 융합
        purp_emb  = self.purpose_emb_shared(purpose_idx)      # (B, emb_dim)
        fused     = torch.cat([feat_repr, purp_emb], dim=-1)  # (B, hidden+emb)
        shared    = self.shared_encoder(fused)                 # (B, hidden)

        # 5) Goal Head
        goal_in    = torch.cat([shared, weighted_sum], dim=-1)
        goal_score = self.goal_head(goal_in).squeeze(-1) * 10.0  # (B,)

        # 6) CEFR Head
        cefr_logits = self.cefr_head(shared)                  # (B, 6)

        return goal_score, cefr_logits, pw
