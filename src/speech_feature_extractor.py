"""
Speech Feature Extractor
========================
실제 음성 파일(.wav)에서 어학 능력 평가에 필요한 피처를 추출.

핵심 설계 원칙 (리팩토링 v2):
  wav2vec2(360MB)와 Whisper(150MB)를 요청마다 로드하면
  서버가 죽을 수 있음 → SpeechFeatureExtractor 클래스의
  __init__에서 딱 한 번만 로드하고 재사용.

  [Before - 문제 있는 구조]
  요청 1 → wav2vec 로드(360MB) → 추출 → 메모리 해제
  요청 2 → wav2vec 로드(360MB) → 추출 → 메모리 해제  ← 느리고 서버 죽음

  [After - 올바른 구조]
  서버 시작 → SpeechFeatureExtractor() → wav2vec 로드(딱 한 번)
  요청 1 → extractor.extract(audio) → 즉시 결과
  요청 2 → extractor.extract(audio) → 즉시 결과  ← 빠름

추출하는 피처 (논문 Bamdev 2023 + GOPT 2022 기반):
  Group A: Pronunciation — wav2vec 2.0 레이어별 임베딩
  Group B: Fluency       — 말하기 속도, 침묵 비율
  Group C: Prosody       — 피치(F0), 에너지(RMS)
  Group D: Completeness  — Whisper ASR 기반 단어 수

실행 방법:
  python src/speech_feature_extractor.py --demo
  python src/speech_feature_extractor.py --audio your_speech.wav --purpose travel
"""

import argparse
import json
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_NAMES = [
    'accuracy', 'prosodic', 'fluency', 'completeness',
    'word_acc_mean', 'word_acc_std', 'word_acc_min', 'word_total_mean',
    'phone_acc_mean', 'phone_acc_std', 'phone_acc_min',
]

PURPOSE_WEIGHTS = {
    'travel'  : {'accuracy': 0.40, 'completeness': 0.10,
                 'fluency' : 0.30, 'prosodic'    : 0.20},
    'business': {'accuracy': 0.30, 'completeness': 0.40,
                 'fluency' : 0.20, 'prosodic'    : 0.10},
    'academic': {'accuracy': 0.20, 'completeness': 0.30,
                 'fluency' : 0.20, 'prosodic'    : 0.30},
}


# ══════════════════════════════════════════════════════════════
# 핵심 클래스: SpeechFeatureExtractor
# ══════════════════════════════════════════════════════════════

class SpeechFeatureExtractor:
    """
    무거운 모델(wav2vec2, Whisper)을 __init__에서 딱 한 번만 로드.
    이후 extract() 호출 시 이미 로드된 모델을 재사용.

    사용 예시 (앱 서버):
        # 서버 시작 시 한 번만 초기화
        extractor = SpeechFeatureExtractor(use_wav2vec=True, use_whisper=True)

        # 요청마다 호출 (모델 재로드 없음 → 빠름)
        features = extractor.extract("user_speech.wav")
    """

    def __init__(
        self,
        use_wav2vec : bool = True,
        use_whisper : bool = True,
        device      : torch.device = None,
    ):
        self.device      = device or DEVICE
        self.use_wav2vec = use_wav2vec
        self.use_whisper = use_whisper

        # wav2vec2 모델 — __init__에서 한 번만 로드
        self._wav2vec_processor = None
        self._wav2vec_model     = None
        if use_wav2vec:
            self._load_wav2vec()

        # Whisper 모델 — __init__에서 한 번만 로드
        self._whisper_model = None
        if use_whisper:
            self._load_whisper()

    def _load_wav2vec(self) -> None:
        """wav2vec2-base 로드 (최초 실행 시 ~360MB 다운로드)."""
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            print("[SpeechFeatureExtractor] Loading wav2vec2-base (~360MB)...")
            self._wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base"
            )
            self._wav2vec_model = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base"
            ).to(self.device)
            self._wav2vec_model.eval()
            print("[SpeechFeatureExtractor] wav2vec2 loaded.")
        except Exception as e:
            print(f"[SpeechFeatureExtractor] wav2vec2 load failed: {e}. "
                  f"Using MFCC fallback.")
            self.use_wav2vec = False

    def _load_whisper(self) -> None:
        """Whisper tiny 로드 (최초 실행 시 ~150MB 다운로드)."""
        try:
            import whisper
            print("[SpeechFeatureExtractor] Loading Whisper tiny (~150MB)...")
            self._whisper_model = whisper.load_model("tiny")
            print("[SpeechFeatureExtractor] Whisper loaded.")
        except Exception as e:
            print(f"[SpeechFeatureExtractor] Whisper load failed: {e}. "
                  f"Using duration fallback.")
            self.use_whisper = False

    # ── 음성 로드 ────────────────────────────────────────────────
    def _load_audio(self, audio_path: str,
                    target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        y, _  = librosa.effects.trim(y, top_db=20)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        return y, sr

    # ── Group A: Pronunciation ───────────────────────────────────
    def _pronunciation(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        발음 정확도 피처.
        논문: Kim et al. (2022) — wav2vec2 중간 레이어 L2 norm이
        발음 정확도와 PCC=0.73+ 상관관계
        """
        if self.use_wav2vec and self._wav2vec_model is not None:
            inputs = self._wav2vec_processor(
                y, sampling_rate=sr, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self._wav2vec_model(
                    **inputs, output_hidden_states=True
                )

            # 중간 레이어(6~9번) 평균 → 발음 관련 표현 집중
            hidden = outputs.hidden_states
            mid    = torch.stack(hidden[6:10], dim=0).mean(0).squeeze(0)
            norms  = mid.norm(dim=-1).cpu().numpy()

            accuracy       = float(np.clip((np.mean(norms) - 30) / 4.0,   0, 10))
            phone_acc_mean = float(np.clip((np.mean(norms) - 25) / 4.5,   0, 2))
            phone_acc_std  = float(np.clip(1.0 / (np.std(norms) + 1e-6) * 0.5, 0, 2))
            phone_acc_min  = float(np.clip((np.percentile(norms,10)-20)/5, 0, 2))

            seg_size = 50
            n_segs   = len(norms) // seg_size
            if n_segs > 0:
                segs  = norms[:n_segs*seg_size].reshape(n_segs, seg_size)
                sm    = segs.mean(axis=1)
                wam   = float(np.clip((sm.mean()-25)/4.0, 0, 10))
                was   = float(np.clip(sm.std()/2.0, 0, 5))
                wamin = float(np.clip((sm.min()-20)/5.0, 0, 10))
            else:
                wam, was, wamin = accuracy, 0.5, accuracy * 0.8

            return {
                'accuracy'       : round(accuracy,       3),
                'word_acc_mean'  : round(wam,             3),
                'word_acc_std'   : round(was,             3),
                'word_acc_min'   : round(wamin,           3),
                'word_total_mean': round(wam,             3),
                'phone_acc_mean' : round(phone_acc_mean,  3),
                'phone_acc_std'  : round(phone_acc_std,   3),
                'phone_acc_min'  : round(phone_acc_min,   3),
            }

        # MFCC fallback
        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        energy = np.sum(mfcc ** 2, axis=0)
        ne     = float(np.clip(np.mean(energy) / 1000, 0, 10))
        return {
            'accuracy'       : round(ne,                              3),
            'word_acc_mean'  : round(ne * 0.95,                       3),
            'word_acc_std'   : round(float(np.std(energy) / 500),     3),
            'word_acc_min'   : round(ne * 0.70,                       3),
            'word_total_mean': round(ne * 0.90,                       3),
            'phone_acc_mean' : round(ne / 5.0,                        3),
            'phone_acc_std'  : round(float(np.std(energy) / 2500),    3),
            'phone_acc_min'  : round(ne / 6.0,                        3),
        }

    # ── Group B: Fluency ─────────────────────────────────────────
    def _fluency(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        유창성 피처.
        논문: Bamdev (2023) — 침묵 비율, 말하기 속도가 feature importance 상위
        """
        duration    = len(y) / sr
        intervals   = librosa.effects.split(y, top_db=25)
        speech_dur  = sum((e - s) for s, e in intervals) / sr
        pause_ratio = 1.0 - (speech_dur / duration) if duration > 0 else 0.5
        zcr         = librosa.feature.zero_crossing_rate(y)[0]
        speech_rate = float(np.clip(np.mean(zcr) * 500, 0, 10))
        pause_score = float(np.clip((1 - pause_ratio) * 10, 0, 10))
        fluency     = float(np.clip(speech_rate * 0.4 + pause_score * 0.6, 0, 10))
        return {
            'fluency'     : round(fluency,      3),
            'speech_rate' : round(speech_rate,  3),
            'pause_ratio' : round(pause_ratio,  3),
            'duration'    : round(duration,     3),
        }

    # ── Group C: Prosody ─────────────────────────────────────────
    def _prosody(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        운율 피처.
        논문: GOPT (Gong 2022) — prosodic = 강세 + 억양
        """
        try:
            f0, voiced, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'), sr=sr
            )
            f0_v       = f0[voiced] if voiced.any() else np.array([150.0])
            pitch_mean = float(np.nanmean(f0_v))
            pitch_std  = float(np.nanstd(f0_v))
        except Exception:
            pitches, mags = librosa.piptrack(y=y, sr=sr)
            vals       = pitches[mags > np.median(mags)]
            pitch_mean = float(np.mean(vals)) if len(vals) > 0 else 150.0
            pitch_std  = float(np.std(vals))  if len(vals) > 0 else 30.0

        rms          = librosa.feature.rms(y=y)[0]
        rms_mean     = float(np.mean(rms))
        pitch_score  = float(np.clip(pitch_std / 30.0, 0, 10))
        energy_score = float(np.clip(rms_mean * 200,   0, 10))
        prosodic     = float(np.clip(pitch_score * 0.5 + energy_score * 0.5, 0, 10))
        return {
            'prosodic'  : round(prosodic,    3),
            'pitch_mean': round(pitch_mean,  3),
            'pitch_std' : round(pitch_std,   3),
            'rms_mean'  : round(rms_mean,    3),
        }

    # ── Group D: Completeness ────────────────────────────────────
    def _completeness(self, y: np.ndarray, sr: int,
                      reference_text: Optional[str] = None) -> Dict[str, float]:
        """
        완성도 피처.
        논문: MultiPA (Chen 2023) — ASR 전사 결과로 완성도 측정
        """
        if self.use_whisper and self._whisper_model is not None:
            import os
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, y, sr)
                tmp = f.name
            result     = self._whisper_model.transcribe(tmp, language='en')
            transcript = result['text'].strip()
            os.unlink(tmp)
            words = transcript.split()
            n     = len(words)
            if reference_text:
                ref_n        = len(reference_text.split())
                completeness = float(np.clip(n / max(ref_n, 1), 0, 1) * 10)
            else:
                ttr          = len(set(w.lower() for w in words)) / max(n, 1)
                completeness = float(np.clip(ttr * 10 + n * 0.1, 0, 10))
            return {'completeness': round(completeness, 3),
                    'transcript'  : transcript, 'n_words': n}

        # duration fallback
        dur          = len(y) / sr
        completeness = float(np.clip(dur / 5.0 * 8, 0, 10))
        return {'completeness': round(completeness, 3),
                'transcript'  : '', 'n_words': 0}

    # ── 통합 추출 (외부에서 호출하는 메서드) ──────────────────────
    def extract(
        self,
        audio_path    : str,
        reference_text: Optional[str] = None,
        verbose       : bool = True,
    ) -> Dict[str, float]:
        """
        음성 파일 → 피처 딕셔너리 반환.
        모델은 이미 __init__에서 로드돼 있으므로 즉시 실행.

        Args:
            audio_path     : 음성 파일 경로
            reference_text : 참조 텍스트 (completeness 정확도 향상)
            verbose        : 상세 출력 여부

        Returns:
            features : FEATURE_NAMES 순서의 딕셔너리
        """
        if verbose:
            print(f"\n  Audio: {audio_path}")

        y, sr = self._load_audio(audio_path)

        if verbose:
            print(f"  Loaded: {len(y)/sr:.2f}s @ {sr}Hz")
            print("  [1/4] Pronunciation...")

        pron = self._pronunciation(y, sr)

        if verbose: print("  [2/4] Fluency...")
        flu  = self._fluency(y, sr)

        if verbose: print("  [3/4] Prosody...")
        pros = self._prosody(y, sr)

        if verbose: print("  [4/4] Completeness...")
        comp = self._completeness(y, sr, reference_text)

        features = {
            'accuracy'       : pron['accuracy'],
            'prosodic'       : pros['prosodic'],
            'fluency'        : flu['fluency'],
            'completeness'   : comp['completeness'],
            'word_acc_mean'  : pron['word_acc_mean'],
            'word_acc_std'   : pron['word_acc_std'],
            'word_acc_min'   : pron['word_acc_min'],
            'word_total_mean': pron['word_total_mean'],
            'phone_acc_mean' : pron['phone_acc_mean'],
            'phone_acc_std'  : pron['phone_acc_std'],
            'phone_acc_min'  : pron['phone_acc_min'],
            '_transcript'    : comp.get('transcript', ''),
            '_duration'      : flu['duration'],
            '_pitch_mean'    : pros['pitch_mean'],
            '_pause_ratio'   : flu['pause_ratio'],
        }

        if verbose:
            print("\n  Extracted Features:")
            print(f"  {'Feature':<20} {'Value':>8}")
            print("  " + "-" * 30)
            for k, v in features.items():
                if not k.startswith('_'):
                    print(f"  {k:<20} {v:>8.3f}")
            if features['_transcript']:
                print(f"\n  ASR Transcript: \"{features['_transcript']}\"")

        return features


# ══════════════════════════════════════════════════════════════
# 하위 호환성: 기존 extract_all_features() 함수 유지
# (infer.py 등 기존 코드가 이 함수를 쓰므로 유지)
# ══════════════════════════════════════════════════════════════

# 모듈 레벨 싱글톤 — 처음 호출 시 한 번만 생성
_default_extractor: Optional[SpeechFeatureExtractor] = None

def extract_all_features(
    audio_path    : str,
    reference_text: Optional[str] = None,
    use_wav2vec   : bool = True,
    verbose       : bool = True,
) -> Dict[str, float]:
    """
    기존 함수형 인터페이스 유지 (하위 호환성).
    내부적으로 SpeechFeatureExtractor 싱글톤을 사용.
    """
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = SpeechFeatureExtractor(
            use_wav2vec = use_wav2vec,
            use_whisper = True,
        )
    return _default_extractor.extract(audio_path, reference_text, verbose)


def generate_demo_audio(path: str = 'demo.wav',
                        duration: float = 3.0,
                        sr: int = 16000) -> str:
    """테스트용 합성 음성 생성 (librosa/soundfile만 사용)."""
    t = np.linspace(0, duration, int(sr * duration))
    y = (0.5 * np.sin(2 * np.pi * 200 * t) +
         0.3 * np.sin(2 * np.pi * 400 * t) +
         0.1 * np.sin(2 * np.pi * 800 * t) +
         0.05 * np.random.randn(len(t)))

    # 단어 경계 근사 (주기적 침묵)
    for i in range(0, len(t), sr // 3):
        y[i: i + sr // 20] *= 0.05

    y = y / np.max(np.abs(y))
    sf.write(path, y, sr)
    print(f"  Demo audio: {path} ({duration:.1f}s)")
    return path


# ══════════════════════════════════════════════════════════════
# 8. 메인
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Extract speech features for language proficiency assessment'
    )
    parser.add_argument('--audio',      type=str, default=None)
    parser.add_argument('--ref',        type=str, default=None,
                        help='Reference text for completeness scoring')
    parser.add_argument('--purpose',    type=str, default='travel',
                        choices=['travel', 'business', 'academic'])
    parser.add_argument('--demo',       action='store_true')
    parser.add_argument('--no-wav2vec', action='store_true')
    parser.add_argument('--output',     type=str, default=None,
                        help='Save features to JSON')
    args = parser.parse_args()

    print("=" * 60)
    print("Speech Feature Extractor")
    print("goal-aware-proficiency project")
    print("=" * 60)

    # 오디오 결정
    demo_path = None
    if args.demo or args.audio is None:
        print("\n[Demo] Generating synthetic audio...")
        demo_path  = 'demo.wav'
        audio_path = generate_demo_audio(demo_path)
    else:
        audio_path = args.audio
        if not Path(audio_path).exists():
            print(f"File not found: {audio_path}")
            return

    # 피처 추출
    features = extract_all_features(
        audio_path     = audio_path,
        reference_text = args.ref,
        use_wav2vec    = not args.no_wav2vec,
        verbose        = True,
    )

    # 목적별 가중 점수
    print(f"\n  Purpose-based Weighted Scores:")
    print(f"  {'Purpose':<12} {'Score':>8}")
    print("  " + "-" * 22)
    for purpose, w in PURPOSE_WEIGHTS.items():
        score = sum(features[f] * w[f]
                    for f in ['accuracy','completeness','fluency','prosodic'])
        mark = " ← selected" if purpose == args.purpose else ""
        print(f"  {purpose:<12} {score:>8.3f}{mark}")

    # JSON 저장
    if args.output:
        out = {k: v for k, v in features.items()}
        out['purpose'] = args.purpose
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved: {args.output}")

    # 데모 파일 정리
    if demo_path and Path(demo_path).exists():
        Path(demo_path).unlink()

    print("\nNext step: pass these features to dual_head_v2.py for scoring.")


if __name__ == '__main__':
    main()
