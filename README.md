# goal-aware-proficiency

> Purpose-conditioned English proficiency assessment model that evaluates learners based on their learning goal (travel, business, academic) alongside absolute CEFR-level scoring.

---

## Overview

Most automated language proficiency systems output a single absolute score (e.g., CEFR A1–C2). However, the **importance of each linguistic dimension varies by learning goal**:

- A **travel learner** needs clear pronunciation and fluency above all
- A **business/job learner** needs formal register and vocabulary completeness
- An **academic learner** needs prosodic control and discourse coherence

This project proposes a **dual-output assessment model** that produces:
1. **Goal-conditioned score** — weighted by the learner's purpose
2. **Absolute proficiency level** — CEFR-aligned absolute score

---

## Key Findings (EDA)

| Feature | Pearson r with Total |
|---|---|
| Accuracy (pronunciation) | **0.9490** |
| Prosodic (rhythm/intonation) | 0.8566 |
| Fluency | 0.8513 |
| Completeness | 0.2076 |

`accuracy` alone accounts for **90.67%** of RandomForest feature importance under equal weighting — motivating purpose-specific reweighting.

---

## Baseline ML Results

| Model | PCC | RMSE | MAE | R² |
|---|---|---|---|---|
| Ridge | **0.9655** | 0.4036 | 0.3514 | 0.9322 |
| GradientBoosting | 0.9652 | 0.4061 | 0.2948 | 0.9313 |
| RandomForest | 0.9602 | 0.4326 | 0.2650 | 0.9220 |
| Lasso | 0.9629 | 0.4349 | 0.3733 | 0.9212 |
| SVR | 0.9466 | 0.5013 | 0.3032 | 0.8953 |

---

## Purpose Weights (Hypothesis)

| Purpose | Accuracy | Completeness | Fluency | Prosodic |
|---|---|---|---|---|
| Equal (baseline) | 0.25 | 0.25 | 0.25 | 0.25 |
| Travel | **0.40** | 0.10 | **0.30** | 0.20 |
| Business/Job | 0.30 | **0.40** | 0.20 | 0.10 |
| Academic | 0.20 | 0.30 | 0.20 | **0.30** |

> Weight rationale is based on EDA correlation analysis and second language acquisition literature. Validation via expert survey is planned.

---

## Project Structure

```
goal-aware-proficiency/
├── notebooks/
│   └── speechocean762_eda.ipynb   # Exploratory data analysis
├── src/
│   └── baseline_ml.py             # Baseline ML model (Bamdev 2023 style)
├── results/
│   ├── baseline_results.png       # Visualization
│   ├── model_results.csv          # Model performance metrics
│   ├── feature_importance.csv     # Feature importance scores
│   └── summary.json               # Full experiment summary
├── requirements.txt
└── README.md
```

---

## Dataset

**speechocean762** (Zhang et al., INTERSPEECH 2021)
- 5,000 English utterances from 250 non-native speakers (L1: Mandarin)
- Annotated by 5 independent experts at sentence / word / phoneme level
- License: CC BY 4.0 (free for commercial and non-commercial use)
- Download: [OpenSLR](https://www.openslr.org/101/) | [HuggingFace](https://huggingface.co/datasets/mispeech/speechocean762)

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/goal-aware-proficiency.git
cd goal-aware-proficiency
pip install -r requirements.txt
```

---

## Usage

### 1. Run EDA
```bash
jupyter notebook notebooks/speechocean762_eda.ipynb
```

### 2. Train Baseline ML Model
```bash
python src/baseline_ml.py
# Results saved to ./results/
```

---

## Roadmap

- [x] Exploratory data analysis (speechocean762)
- [x] Baseline ML model (Ridge / SVR / RandomForest / GradientBoosting)
- [x] Feature importance analysis (RF importance + Ridge coefficients)
- [x] Purpose-based weighted score simulation
- [ ] Purpose-conditioned dual-head model
- [ ] Formality/register detection module (Formality-BERT)
- [ ] CEFR absolute scoring head (distilBERT fine-tuning on Ace-CEFR)
- [ ] Validation of purpose weights (expert survey)
- [ ] Integration & evaluation

---

## References

```bibtex
@article{bamdev2023,
  title   = {Automated Speech Scoring System Under The Lens},
  author  = {Bamdev, Pakhi and Grover, Manraj Singh and others},
  journal = {International Journal of Artificial Intelligence in Education},
  volume  = {33},
  pages   = {119--154},
  year    = {2023}
}

@inproceedings{zhang2021speechocean762,
  title     = {speechocean762: An Open-Source Non-native English Speech Corpus
               For Pronunciation Assessment},
  author    = {Zhang, Junbo and others},
  booktitle = {Proc. INTERSPEECH},
  year      = {2021}
}

@inproceedings{gong2022gopt,
  title     = {Transformer-based Multi-Aspect Multi-Granularity Non-Native
               English Speaker Pronunciation Assessment},
  author    = {Gong, Yuan and others},
  booktitle = {ICASSP},
  year      = {2022}
}
```

---

## License

MIT License
