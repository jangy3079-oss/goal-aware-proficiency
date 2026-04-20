"""
infer.py
========
End-to-End 추론 파이프라인.

음성 파일 → 피처 추출 → 모델 → 점수 출력

이 파일이 완성되면:
  실제 앱에서 사용자가 음성을 녹음하면
  목적 맞춤 점수 + CEFR 레벨을 바로 출력할 수 있음.

사용법:
  # 데모 (합성 음성)
  python src/infer.py --demo --purpose travel

  # 실제 음성 파일
  python src/infer.py --audio my_speech.wav --purpose business

  # 참조 텍스트 있을 때 (completeness 더 정확)
  python src/infer.py --audio my_speech.wav --purpose academic --ref "I would like to present"

파이프라인:
  [.wav 파일]
      ↓ speech_feature_extractor.py
  [피처 11개] (accuracy, fluency, prosodic ...)
      ↓ model.py (DualHeadModelV2)
      ↓ + [목적 선택: travel/business/academic]
  [목적 맞춤 점수 0~10] + [CEFR 레벨 A1~C2]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml

warnings.filterwarnings('ignore')

# src/ 폴더 내 모듈 import
sys.path.insert(0, str(Path(__file__).parent))
from model import DualHeadModelV2
from speech_feature_extractor import extract_all_features, generate_demo_audio

CEFR_LABELS  = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
PURPOSES     = ['travel', 'business', 'academic']
PURPOSE2IDX  = {p: i for i, p in enumerate(PURPOSES)}
FEATURE_COLS = [
    'accuracy', 'prosodic', 'fluency', 'completeness',
    'word_acc_mean', 'word_acc_std', 'word_acc_min', 'word_total_mean',
    'phone_acc_mean', 'phone_acc_std', 'phone_acc_min',
]

# 목적별 설명 (앱 피드백용)
PURPOSE_DESCRIPTIONS = {
    'travel': {
        'focus'   : 'Pronunciation accuracy & fluency',
        'tip'     : 'For travel, clear pronunciation and smooth speech flow are most important.',
        'key_feat': 'accuracy',
    },
    'business': {
        'focus'   : 'Vocabulary completeness & accuracy',
        'tip'     : 'For business, using complete and precise vocabulary matters most.',
        'key_feat': 'completeness',
    },
    'academic': {
        'focus'   : 'Prosodic control & completeness',
        'tip'     : 'For academic use, natural intonation and stress patterns are key.',
        'key_feat': 'prosodic',
    },
}


# ══════════════════════════════════════════════════════════════
# 1. 모델 로드
# ══════════════════════════════════════════════════════════════

class ProficiencyInferencer:
    """
    학습된 모델을 로드하고 음성 파일에서 점수를 추론하는 클래스.

    앱에서는 이 클래스를 한 번 초기화한 뒤
    predict() 메서드를 반복 호출하는 방식으로 사용.
    """

    def __init__(
        self,
        checkpoint_path: str = 'results/dualhead_v2.pth',
        config_path    : str = 'src/config.yaml',
        device         : Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # config 로드
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        # 체크포인트 로드
        print(f"[Inferencer] Loading model from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # 모델 초기화 & 가중치 로드
        self.model = DualHeadModelV2(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        # 정규화 scaler 복원
        from sklearn.preprocessing import StandardScaler
        self.scaler       = StandardScaler()
        self.scaler.mean_ = np.array(ckpt['scaler_mean'])
        self.scaler.scale_= np.array(ckpt['scaler_scale'])
        # sklearn scaler가 transform 가능하도록 n_features_in_ 설정
        self.scaler.n_features_in_ = len(FEATURE_COLS)

        print(f"[Inferencer] Model loaded. Device: {self.device}")

    @torch.no_grad()
    def predict(
        self,
        audio_path    : str,
        purpose       : str = 'travel',
        reference_text: Optional[str] = None,
        use_wav2vec   : bool = False,   # Mac CPU에서는 False 권장 (속도)
        verbose       : bool = True,
    ) -> Dict:
        """
        음성 파일 → 목적 맞춤 점수 + CEFR 레벨 출력.

        Args:
            audio_path     : 음성 파일 경로 (.wav/.mp3/.flac)
            purpose        : 학습자가 선택한 목적 (travel/business/academic)
            reference_text : 참조 텍스트 (completeness 정확도 향상)
            use_wav2vec    : True=wav2vec2 사용, False=MFCC 빠른 모드
            verbose        : 상세 출력 여부

        Returns:
            result: {
                'goal_score'     : float  — 목적 맞춤 점수 (0~10)
                'cefr_level'     : str    — CEFR 레벨 (A1~C2)
                'cefr_idx'       : int    — CEFR 인덱스 (0~5)
                'purpose_weights': dict   — 학습된 목적별 가중치
                'features'       : dict   — 추출된 피처
                'feedback'       : str    — 자연어 피드백
            }
        """
        if purpose not in PURPOSE2IDX:
            raise ValueError(f"purpose must be one of {PURPOSES}")

        # ── Step 1: 음성 → 피처 추출 ─────────────────────────────
        if verbose:
            print(f"\n[Step 1] Extracting features from audio...")

        raw_features = extract_all_features(
            audio_path     = audio_path,
            reference_text = reference_text,
            use_wav2vec    = use_wav2vec,
            verbose        = verbose,
        )

        # ── Step 2: 피처 벡터 구성 ───────────────────────────────
        feat_vec = np.array(
            [[raw_features[col] for col in FEATURE_COLS]],
            dtype=np.float32
        )   # (1, 11)

        # ── Step 3: 정규화 ───────────────────────────────────────
        feat_scaled = self.scaler.transform(feat_vec)
        feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(self.device)

        # sent_feats: accuracy/completeness/fluency/prosodic (인덱스 0,3,2,1)
        sent_tensor = feat_tensor[:, [0, 3, 2, 1]]

        # 목적 인덱스
        pidx = torch.tensor(
            [PURPOSE2IDX[purpose]], dtype=torch.long
        ).to(self.device)

        # ── Step 4: 모델 추론 ────────────────────────────────────
        if verbose:
            print(f"\n[Step 2] Running model inference...")

        goal_score, cefr_logits, pw = self.model(
            feat_tensor, pidx, sent_tensor
        )

        goal_val  = float(goal_score.cpu().item())
        cefr_idx  = int(cefr_logits.argmax(-1).cpu().item())
        cefr_label= CEFR_LABELS[cefr_idx]
        pw_np     = pw.squeeze().cpu().numpy()

        purpose_weights = {
            feat: round(float(w), 4)
            for feat, w in zip(
                ['accuracy','completeness','fluency','prosodic'], pw_np
            )
        }

        # ── Step 5: 피드백 생성 ──────────────────────────────────
        feedback = self._generate_feedback(
            purpose, goal_val, cefr_label, raw_features, purpose_weights
        )

        result = {
            'goal_score'     : round(goal_val, 3),
            'cefr_level'     : cefr_label,
            'cefr_idx'       : cefr_idx,
            'purpose'        : purpose,
            'purpose_weights': purpose_weights,
            'raw_features'   : {
                k: v for k, v in raw_features.items()
                if not k.startswith('_')
            },
            'transcript'     : raw_features.get('_transcript', ''),
            'feedback'       : feedback,
        }

        return result

    def _generate_feedback(
        self,
        purpose         : str,
        goal_score      : float,
        cefr_label      : str,
        features        : Dict,
        purpose_weights : Dict,
    ) -> str:
        """
        점수 기반 자연어 피드백 생성.
        추후 LLM API 연동 시 이 부분을 교체하면 더 풍부한 피드백 가능.
        """
        desc     = PURPOSE_DESCRIPTIONS[purpose]
        key_feat = desc['key_feat']
        key_val  = features.get(key_feat, 0)

        # 점수 구간별 평가
        if goal_score >= 8.0:
            level_msg = "Excellent"
        elif goal_score >= 6.0:
            level_msg = "Good"
        elif goal_score >= 4.0:
            level_msg = "Fair"
        else:
            level_msg = "Needs improvement"

        # 핵심 피처 약점 파악
        weakest = min(
            ['accuracy','completeness','fluency','prosodic'],
            key=lambda f: features.get(f, 0) * purpose_weights.get(f, 0.25)
        )

        feedback = (
            f"[{purpose.upper()} PURPOSE]\n"
            f"Overall: {level_msg} (Score: {goal_score:.1f}/10, CEFR: {cefr_label})\n"
            f"Focus area: {desc['focus']}\n"
            f"Tip: {desc['tip']}\n"
            f"Suggested improvement: Work on '{weakest}' "
            f"(current: {features.get(weakest, 0):.2f}/10)"
        )

        return feedback

    def predict_all_purposes(
        self,
        audio_path    : str,
        reference_text: Optional[str] = None,
        use_wav2vec   : bool = False,
        verbose       : bool = True,
    ) -> Dict[str, Dict]:
        """
        같은 음성에 대해 3가지 목적 모두 평가.
        논문 Table 3 (Inference Demo) 생성용.
        """
        # 피처는 한 번만 추출 (속도 최적화)
        if verbose:
            print("\n[Extracting features once for all purposes...]")

        raw_features = extract_all_features(
            audio_path     = audio_path,
            reference_text = reference_text,
            use_wav2vec    = use_wav2vec,
            verbose        = verbose,
        )

        feat_vec    = np.array(
            [[raw_features[col] for col in FEATURE_COLS]], dtype=np.float32
        )
        feat_scaled = self.scaler.transform(feat_vec)
        feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(self.device)
        sent_tensor = feat_tensor[:, [0, 3, 2, 1]]

        results = {}
        with torch.no_grad():
            for purpose in PURPOSES:
                pidx = torch.tensor(
                    [PURPOSE2IDX[purpose]], dtype=torch.long
                ).to(self.device)

                goal_score, cefr_logits, pw = self.model(
                    feat_tensor, pidx, sent_tensor
                )

                goal_val  = float(goal_score.cpu().item())
                cefr_idx  = int(cefr_logits.argmax(-1).cpu().item())
                pw_np     = pw.squeeze().cpu().numpy()

                results[purpose] = {
                    'goal_score'     : round(goal_val, 3),
                    'cefr_level'     : CEFR_LABELS[cefr_idx],
                    'purpose_weights': {
                        f: round(float(w), 4)
                        for f, w in zip(
                            ['accuracy','completeness','fluency','prosodic'], pw_np
                        )
                    },
                }

        return results, raw_features


# ══════════════════════════════════════════════════════════════
# 2. 결과 출력 포맷
# ══════════════════════════════════════════════════════════════

def print_single_result(result: Dict) -> None:
    """단일 목적 결과 출력."""
    print("\n" + "=" * 60)
    print("ASSESSMENT RESULT")
    print("=" * 60)
    print(f"  Purpose    : {result['purpose'].upper()}")
    print(f"  Goal Score : {result['goal_score']:.2f} / 10.0")
    print(f"  CEFR Level : {result['cefr_level']}")
    if result['transcript']:
        print(f"  Transcript : \"{result['transcript']}\"")
    print(f"\n  Learned Weights (how this purpose values each feature):")
    for feat, w in result['purpose_weights'].items():
        bar = '█' * int(w * 30)
        print(f"    {feat:<14}: {w:.3f}  {bar}")
    print(f"\n  Feedback:\n")
    for line in result['feedback'].split('\n'):
        print(f"    {line}")


def print_all_purposes_result(
    results     : Dict[str, Dict],
    raw_features: Dict,
) -> None:
    """3가지 목적 비교 출력 — 논문 Table 3 형태."""
    print("\n" + "=" * 62)
    print("ALL PURPOSES COMPARISON (논문 Table 3 형태)")
    print("Same speech, different purpose → different score")
    print("=" * 62)

    print(f"\n  Raw Features:")
    for feat in ['accuracy','completeness','fluency','prosodic']:
        print(f"    {feat:<16}: {raw_features.get(feat, 0):.3f}")

    print(f"\n  {'Purpose':<12} {'Goal Score':>11} {'CEFR':>6}  "
          f"Weights (acc/comp/flu/pros)")
    print("  " + "-" * 62)

    for purpose, res in results.items():
        w = res['purpose_weights']
        print(
            f"  {purpose:<12} {res['goal_score']:>11.3f} "
            f"{res['cefr_level']:>6}  "
            f"[{w['accuracy']:.3f} / {w['completeness']:.3f} / "
            f"{w['fluency']:.3f} / {w['prosodic']:.3f}]"
        )

    # 점수 차이가 가장 큰 두 목적 하이라이트
    scores = {p: r['goal_score'] for p, r in results.items()}
    best   = max(scores, key=scores.get)
    worst  = min(scores, key=scores.get)
    diff   = scores[best] - scores[worst]
    print(f"\n  Highest: {best} ({scores[best]:.3f})")
    print(f"  Lowest : {worst} ({scores[worst]:.3f})")
    print(f"  Gap    : {diff:.3f} pts — "
          f"{'significant difference' if diff > 0.5 else 'similar across purposes'}")


# ══════════════════════════════════════════════════════════════
# 3. 메인
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='End-to-End speech proficiency assessment'
    )
    parser.add_argument('--audio',      type=str,    default=None)
    parser.add_argument('--purpose',    type=str,    default='travel',
                        choices=PURPOSES)
    parser.add_argument('--ref',        type=str,    default=None)
    parser.add_argument('--all',        action='store_true',
                        help='Evaluate all 3 purposes at once')
    parser.add_argument('--demo',       action='store_true',
                        help='Use synthetic demo audio')
    parser.add_argument('--no-wav2vec', action='store_true',
                        help='Use MFCC fallback (faster)')
    parser.add_argument('--checkpoint', type=str,
                        default='results/dualhead_v2.pth')
    parser.add_argument('--config',     type=str,
                        default='src/config.yaml')
    parser.add_argument('--output',     type=str,    default=None,
                        help='Save result to JSON')
    args = parser.parse_args()

    print("=" * 62)
    print("Goal-Aware Proficiency Inferencer")
    print("End-to-End: Audio → Features → Score")
    print("=" * 62)

    # 체크포인트 존재 확인
    if not Path(args.checkpoint).exists():
        print(f"\n[Error] Checkpoint not found: {args.checkpoint}")
        print("Run 'python src/train.py' first to train the model.")
        return

    # 모델 로드
    inferencer = ProficiencyInferencer(
        checkpoint_path = args.checkpoint,
        config_path     = args.config,
    )

    # 오디오 결정
    demo_path = None
    if args.demo or args.audio is None:
        print("\n[Demo] Generating synthetic audio...")
        demo_path  = 'demo_infer.wav'
        audio_path = generate_demo_audio(demo_path)
    else:
        audio_path = args.audio
        if not Path(audio_path).exists():
            print(f"[Error] Audio file not found: {audio_path}")
            return

    use_wav2vec = not args.no_wav2vec

    # 추론
    if args.all:
        # 3가지 목적 모두 평가
        results, raw_features = inferencer.predict_all_purposes(
            audio_path     = audio_path,
            reference_text = args.ref,
            use_wav2vec    = use_wav2vec,
            verbose        = True,
        )
        print_all_purposes_result(results, raw_features)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump({'results': results,
                           'raw_features': {
                               k: v for k, v in raw_features.items()
                               if not k.startswith('_')
                           }}, f, indent=2)
            print(f"\nSaved: {args.output}")

    else:
        # 단일 목적 평가
        result = inferencer.predict(
            audio_path     = audio_path,
            purpose        = args.purpose,
            reference_text = args.ref,
            use_wav2vec    = use_wav2vec,
            verbose        = True,
        )
        print_single_result(result)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved: {args.output}")

    # 데모 파일 정리
    if demo_path and Path(demo_path).exists():
        Path(demo_path).unlink()

    print("\nDone.")


if __name__ == '__main__':
    main()
