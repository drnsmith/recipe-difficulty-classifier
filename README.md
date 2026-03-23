# Recipe Difficulty Classification: Large-Scale NLP Benchmark

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An end-to-end NLP classification pipeline for recipe difficulty prediction, trained and evaluated on the **RecipeNLG corpus (2.2M recipes)**. This project benchmarks 10 machine learning and deep learning architectures, implements domain-informed feature engineering, and applies LIME explainability to understand model decision-making at the instance level.

---

## Overview

Recipe difficulty labelling is a non-trivial multi-class NLP problem: difficulty is not explicit in the data and must be engineered from structural signals within the recipe text. This project addresses that challenge end to end — from label construction through model selection to explainability — on one of the largest publicly available recipe datasets.

The pipeline is designed to be reproducible, extensible, and connected to a broader recipe intelligence platform currently in development, which adds semantic retrieval, LLM-based generation, and cross-modal image understanding.

---

## Dataset

**RecipeNLG** — 2,231,142 cooking recipes scraped from cookbooks, food blogs, and recipe websites. Published at the 13th International Conference on Natural Language Generation (2020), Poznań University of Technology.

Each recipe contains a title, structured ingredient list, and free-text preparation instructions. Difficulty labels are not provided — they are engineered from the recipe content itself (see label assignment below).

---

## Label Assignment

Difficulty levels are programmatically assigned using a weighted complexity score derived from three domain-informed signals:

| Signal | Description |
|---|---|
| Step complexity | Average word count per preparation step |
| Technique diversity | Count of distinct cooking techniques, normalised across the corpus |
| Ingredient diversity | Number of unique ingredients per recipe |

These signals are normalised and combined into a single **Total Complexity Score**. Recipes are then binned into four difficulty levels — Easy, Medium, Hard, Very Hard — using corpus-wide quantiles to ensure balanced class distribution.

This methodology is consistent with active learning approaches used in published recipe classification research, including the 3A2M corpus (Sakib et al., MIET 2022).

---

## Pipeline

```
RecipeNLG corpus (2.2M recipes)
        │
        ▼
Text pre-processing
  · Lowercasing, punctuation removal
  · Tokenisation and lemmatisation
  · Stopword filtering
        │
        ▼
Feature engineering
  · TF-IDF (baseline representation)
  · Word2Vec (semantic ingredient embeddings)
  · BERT (contextual representations)
  · Complexity score features
        │
        ▼
Model training and evaluation (10 architectures)
        │
        ▼
LIME explainability on best model
        │
        ▼
Difficulty prediction: Easy · Medium · Hard · Very Hard
```

---

## Model Benchmark

Ten architectures evaluated across accuracy, precision, recall, and F1 on a held-out test set:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Baseline (dummy) | 0.250 | — | — | — |
| Naive Bayes | 0.449 | 0.468 | 0.449 | 0.441 |
| Random Forest | 0.636 | 0.630 | 0.636 | 0.632 |
| SVM | 0.667 | 0.668 | 0.667 | 0.667 |
| Gradient Boosting (XGBoost) | 0.684 | 0.685 | 0.684 | 0.684 |
| CNN | 0.690 | 0.693 | 0.690 | 0.691 |
| Logistic Regression | 0.694 | 0.696 | 0.694 | 0.695 |
| RNN | 0.682 | 0.681 | 0.682 | 0.681 |
| MLP | 0.718 | 0.717 | 0.718 | 0.717 |
| Custom Neural Network (v1) | 0.752 | 0.754 | 0.752 | 0.753 |
| **Custom Neural Network (v2)** | **0.750** | **0.757** | **0.750** | **0.753** |

The custom neural network achieves the strongest F1 score (0.753), representing a **+50 percentage point improvement** over the dummy baseline on a 4-class problem — a meaningful result given the inherent ambiguity of difficulty labelling from text alone.

---

## Explainability with LIME

LIME (Local Interpretable Model-agnostic Explanations) is applied to the best-performing model to understand which textual features drive individual predictions. This goes beyond aggregate metrics to surface why the model classifies a specific recipe as it does.

**Example prediction:**

| Class | Probability |
|---|---|
| Easy | 80% |
| Medium | 0% |
| Hard | 20% |
| Very Hard | 0% |

LIME identified influential tokens including `mixtury`, `ingredi`, `side`, `tomato`, and `crumb` as the strongest contributors to this prediction — reflecting that ingredient diversity and preparation complexity are the dominant signals, consistent with the feature engineering design.

This explainability layer is directly relevant to production deployment: a difficulty classifier used in a real cooking assistant needs to be auditable, not just accurate.

---

## Project Structure

```
recipe-difficulty-classifier/
├── AI_Class.ipynb              # Full pipeline: preprocessing → training → LIME
├── data/
│   └── sample.csv              # Sample of RecipeNLG for reproducibility
├── outputs/
│   ├── model_comparison.png    # Benchmark visualisation
│   └── lime_explanation.png    # LIME output for example prediction
└── README.md
```

---

## Part of a Larger System

This classifier is one component of a larger recipe intelligence platform in development, which includes:

- Semantic recipe retrieval over the full 2.2M corpus using sentence-transformer embeddings and FAISS
- LLM-powered recipe generation and adaptation (Llama3)
- Cross-modal image-recipe retrieval using CLIP and EfficientNet fine-tuned on the Recipe1M image dataset (800k+ food images)
- FastAPI backend with Redis caching and Prometheus monitoring
- Streamlit dashboard for interactive exploration

The difficulty classifier feeds into this pipeline as the labelling and filtering layer, enabling difficulty-aware retrieval and generation.

---

## Quickstart

```bash
git clone https://github.com/drnsmith/AI-Recipe-Classifier.git
cd AI-Recipe-Classifier
pip install -r requirements.txt
jupyter notebook AI_Class.ipynb
```

The notebook is self-contained and runs on a sample of the RecipeNLG data. To run on the full 2.2M corpus, download the dataset from [recipenlg.cs.put.poznan.pl](https://recipenlg.cs.put.poznan.pl/) and update the data path in the notebook.

---

## References

- Bień et al. (2020). RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation. *INLG 2020*.
- Sakib et al. (2022). Assorted, Archetypal and Annotated Two Million (3A2M) Cooking Recipes Dataset based on Active Learning. *MIET 2022*.
- Ribeiro et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD 2016*.

---

## Credits

Built by [@drnsmith](https://github.com/drnsmith) as part of a production-oriented data science portfolio.

Technical writing on NLP and applied ML: [Medium](https://medium.com/@NeverOblivious) · [Substack](https://substack.com/@errolog)
