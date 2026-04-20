"""
dataset.py
==========
데이터 로드, 전처리, PyTorch Dataset 클래스 정의.
dual_head_v2.py에서 데이터 관련 코드만 분리.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


# ── CEFR 변환 ──────────────────────────────────────────────────
def score_to_cefr(score: float, thresholds: List[float] = None) -> int:
    """
    총점(0~10) → CEFR 레벨 인덱스(0~5).
    thresholds: config의 cefr.thresholds 값
    """
    if thresholds is None:
        thresholds = [2.0, 4.0, 6.0, 7.5, 9.0]
    for level, t in enumerate(thresholds):
        if score <= t:
            return level
    return len(thresholds)  # C2


# ── 데이터 로드 ─────────────────────────────────────────────────
def load_speechocean(cfg: dict, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    speechocean762 로드 후 목적별 확장(augmentation).

    핵심 설계:
      - 목적 레이블 자동 부여 없음 (v2 핵심)
      - 각 샘플을 3가지 목적으로 복제
      - purpose_idx = 학습자가 앱에서 직접 선택하는 값

    Args:
        cfg: config.yaml 전체 딕셔너리
    Returns:
        train_df, test_df: 확장된 DataFrame
    """
    if verbose:
        print("[Data] Loading speechocean762 from HuggingFace...")

    dataset   = load_dataset(cfg['data']['dataset'])
    feat_cols = cfg['data']['feature_cols']
    sent_cols = cfg['data']['sentence_feats']
    purposes  = cfg['purposes']['list']
    pw        = cfg['purposes']['weights']
    thresholds= cfg['cefr']['thresholds']
    purpose2idx = {p: i for i, p in enumerate(purposes)}

    def to_base_df(split) -> pd.DataFrame:
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

    def expand_by_purpose(df: pd.DataFrame) -> pd.DataFrame:
        """각 샘플을 3가지 목적으로 복제 → 완벽한 균형 분포 보장."""
        rows = []
        for _, row in df.iterrows():
            cefr = score_to_cefr(row['total'], thresholds)
            base = row.to_dict()
            base['cefr_level'] = cefr
            for p in purposes:
                r = base.copy()
                r['purpose']     = p
                r['purpose_idx'] = purpose2idx[p]
                r['goal_score']  = sum(row[f] * pw[p][f] for f in sent_cols)
                rows.append(r)
        return pd.DataFrame(rows)

    base_train = to_base_df(dataset['train'])
    base_test  = to_base_df(dataset['test'])

    train_df = expand_by_purpose(base_train)
    test_df  = expand_by_purpose(base_test)

    if verbose:
        print(f"  Train: {len(base_train)} → {len(train_df)} (×3 purposes)")
        print(f"  Test : {len(base_test)}  → {len(test_df)}")
        print(f"  Purpose distribution: perfectly balanced (33.3% each)")
        cefr_labels = cfg['cefr']['labels']
        print(f"  CEFR distribution (base train):")
        for lvl, cnt in base_train['total'].apply(
                lambda s: score_to_cefr(s, thresholds)).value_counts().sort_index().items():
            print(f"    {cefr_labels[lvl]}: {cnt}")

    return train_df, test_df


# ── PyTorch Dataset ─────────────────────────────────────────────
class ProficiencyDataset(Dataset):
    """
    목적 조건부 어학 능력 평가 데이터셋.

    학습자가 앱에서 purpose_idx를 직접 선택 →
    해당 목적의 가중 점수(goal_score)와 절대 레벨(cefr_level)을 함께 예측.
    """

    def __init__(
        self,
        df          : pd.DataFrame,
        feature_cols: List[str],
        scaler      : StandardScaler = None,
        fit_scaler  : bool = False,
    ):
        X = df[feature_cols].values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            assert scaler is not None, "scaler must be provided when fit_scaler=False"
            self.scaler = scaler
            X = self.scaler.transform(X)

        self.X           = torch.tensor(X,                         dtype=torch.float32)
        self.purpose_idx = torch.tensor(df['purpose_idx'].values,  dtype=torch.long)
        self.goal_score  = torch.tensor(df['goal_score'].values,   dtype=torch.float32)
        self.cefr_level  = torch.tensor(df['cefr_level'].values,   dtype=torch.long)
        self.total_score = torch.tensor(df['total'].values,        dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'features'   : self.X[idx],
            'purpose_idx': self.purpose_idx[idx],
            'goal_score' : self.goal_score[idx],
            'cefr_level' : self.cefr_level[idx],
            'total_score': self.total_score[idx],
        }
