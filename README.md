# Latent Probing for Mental Health Sentiment Classification

**Team:** Rwenzori

## Overview

This project implements a latent probing pipeline to classify mental health sentiment
(Depression, Anxiety, Stress, or Normal) from short user-generated text. Instead of
fine-tuning a large language model end-to-end, we freeze **Gemma** and train lightweight
linear probes on its internal layer representations, then evaluate which layer best
encodes mental health sentiment. This keeps the approach computationally light while
remaining interpretable, in line with the transparency value in our Problem Statement.

## Dataset

We use the [Sentiment Analysis for Mental Health dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
(Kaggle), an aggregation of public, pre-labeled Reddit and Twitter/X posts. Full detail
on sourcing, consent limitations, and bias/representation gaps is documented in our
[Data Card](docs/data_card.pdf).

From the full dataset, we selected four non-crisis distress classes relevant to early
identification:

| Class | Count (full dataset) |
|---|---|
| Normal | 16,343 |
| Depression | 15,404 |
| Anxiety | 3,841 |
| Stress | 2,587 |

Rows with missing or empty text were dropped, and text was cleaned (URLs removed,
whitespace collapsed). Data was split 70/15/15 into train/validation/test, stratified
by label, with the test set touched only once for final evaluation.

## Training Pipeline

1. **Frozen embedding extraction:** Each cleaned post is passed through `google/gemma-2b`
   (frozen, no fine-tuning). Since Gemma is decoder-only and has no `[CLS]` token, we
   mean-pool each layer's hidden states over the real (non-padding) tokens, producing one
   fixed-size vector per layer, per example.
2. **Multi-layer probing:** We repeat this for every layer of the model (embedding layer
   plus all transformer blocks), keeping every layer's representation rather than
   committing to a single one.
3. **Linear probes:** For each layer, we train a separate logistic regression classifier
   (`class_weight="balanced"`, `C=1.0`) directly on that layer's embeddings.
4. **Layer selection:** We compare validation accuracy across all layers to identify which
   depth of the model best encodes mental health sentiment. See `scripts/layer_accuracy_plot.png`
   for the full comparison.
5. **Final model:** The best-performing layer's probe is refit and evaluated once on the
   held-out test set.

Key hyperparameters: `MAX_LENGTH=128` tokens, `BATCH_SIZE=8`, `RANDOM_STATE=42`.
This run used `SAMPLE_FRAC=0.5` of the cleaned dataset (~19,000 rows) to balance
result quality against Colab's free-tier GPU time limits.

## Evaluation

**Best layer:** 13 (validation accuracy: 0.902, train/validation gap: 0.098 — within the
0.15 threshold we use to flag overfitting).

**Final test set results:**

- **Test accuracy: 0.905**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Anxiety | 0.77 | 0.83 | 0.80 | 288 |
| Depression | 0.93 | 0.94 | 0.94 | 1,156 |
| Normal | 0.96 | 0.93 | 0.94 | 1,226 |
| Stress | 0.64 | 0.69 | 0.66 | 194 |
| **Accuracy** | | | **0.91** | 2,864 |
| Macro avg | 0.83 | 0.85 | 0.83 | 2,864 |
| Weighted avg | 0.91 | 0.91 | 0.91 | 2,864 |

**Interpretation:** The model performs strongly on Depression and Normal (94% F1 each),
the two largest classes. Anxiety (80% F1) and especially Stress (66% F1) perform
noticeably weaker, consistent with their smaller representation in the training data
(3,841 and 2,587 examples respectively, versus 15,404–16,343 for Depression/Normal).
The gap between macro avg (0.83) and weighted avg (0.91) reflects this imbalance: overall
accuracy is pulled up by the larger classes. This limitation is discussed further in our
[Data Card's](docs/data_card.pdf) Bias and Representation section, and mitigating it
(e.g. through class rebalancing or additional data for minority classes) is noted as
future work.

The train/validation gap of ~0.10 at the best layer indicates mild overfitting, likely
due to the high dimensionality of Gemma's embeddings relative to training set size —
expected behavior for a linear probe on large frozen representations, and within our
project's stated tolerance.

## Reproduction

To reproduce these results:

1. Open `scripts/mental_health_latent_probing_gemma.ipynb` in Google Colab.
2. Set the runtime to a GPU (Runtime → Change runtime type → T4 GPU).
3. Create a free Hugging Face account, accept Gemma's license at
   [huggingface.co/google/gemma-2b](https://huggingface.co/google/gemma-2b), and
   generate a Read access token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Download `Combined Data.csv` from the
   [Kaggle dataset page](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
   and upload it into the Colab session.
5. Run all cells top to bottom. When prompted, paste your Hugging Face token to
   authenticate and download Gemma.
6. Set `SAMPLE_FRAC` in the config cell (0.5 was used for the reported results above;
   1.0 uses the full dataset but takes proportionally longer).
7. The notebook saves `probe_model.pkl`, `label_encoder.pkl`, `best_layer.txt`, and
   `layer_accuracy_plot.png` at the end of the run.

## Appendix

**Contributors:** Rwenzori team, AI Saturdays Cohort 10.

**Mentors:** Dr. (Mrs.) O. O. Coker

**Repository structure:**
```
C10-team-rwenzori/
├── README.md
├── docs/
│   ├── problem_statement.pdf
│   ├── data_card.pdf
│   ├── impact_statement_card.pdf
│   └── stakeholder_engagement.pdf
└── scripts/
    ├── mental_health_latent_probing_gemma.ipynb
    └── layer_accuracy_plot.png
```
