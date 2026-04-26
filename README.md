# 🌱 Agri-AI-G×E: AI-Driven Genotype × Environment Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agri-ai-gee-mlmodel.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🔬 Overview
An AI-integrated framework that combines **classical quantitative genetics** (BLUP, AMMI/GGE) with **machine learning** (XGBoost) to predict crop yield under varying environmental conditions and identify stable genotypes for climate-stressed breeding programs.

**Key Innovation:** Classical breeding statistics (BLUPs, G×E principal components) serve as engineered features for ML models, creating a *biology-guided AI* that outperforms pure data-driven approaches.

## 📊 Dataset
**USDA SoyURT** (Uniform Regional Soybean Trials) — 39,006 observations across 4,257 genotypes, 63 locations, and 31 years (1989–2019). Licensed under CC BY 4.0.

*Krause, M.D. et al. (2022). Using large soybean historical data to study genotype by environment variation. bioRxiv.*

## 🧠 Methodology
| Layer | Method | Purpose |
|-------|--------|---------|
| **Classical** | BLUP (lme4) | Extract genotype breeding values |
| **Classical** | GGE Biplot (SVD) | Quantify G×E interaction |
| **ML Baseline** | Linear Regression, Random Forest | Performance benchmarks |
| **Hybrid ML** | XGBoost with classical features | Main prediction model |

## 📈 Results
- **Best Model R²:** 0.71 on held-out temporal test set
- **SHAP Analysis:** Confirmed that BLUP and G×E PCs are top predictive features
- **Heritability (H²):** 0.24 — confirming strong environmental influence

## 🚀 Live Demo
**[Launch Interactive Dashboard →](https://agri-ai-gee-mlmodel.streamlit.app/)**

## 📁 Project Structure
agri-ai-gee/
├── data/
│   ├── raw/              # Raw soybean phenotype data
│   └── data_dictionary.md
├── notebooks/
│   ├── EDA.ipynb         # Exploratory data analysis
│   └── modeling.ipynb    # ML pipeline & evaluation
├── src/
│   ├── blup_model.R      # Mixed model for breeding values
│   ├── ammi_model.R      # G×E decomposition via SVD/GGE
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── ml_models.py
│   └── evaluation.py
├── app/
│   └── app.py            # Streamlit dashboard
├── results/
│   ├── figures/          # Publication-quality plots
│   └── tables/           # Model comparison results
└── paper/                # Manuscript draft