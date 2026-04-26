# An AI-Integrated Framework for Genotype × Environment Interaction Modeling and Yield Prediction in Climate-Stressed Crops

**Zulqarnain***
*Independent Researcher | GitHub: [@agristattools](https://github.com/agristattools)*

---

## Abstract

**Background:** Climate change increasingly threatens crop production, making accurate prediction of genotype performance across environments essential for breeding programs. While classical quantitative genetics (BLUP, AMMI) excels at estimating breeding values, and machine learning captures non-linear environmental responses, no framework effectively integrates both approaches.

**Methods:** We developed a hybrid AI-G×E system combining classical mixed models (BLUP) and GGE biplot decomposition with XGBoost machine learning. Classical features (BLUPs, G×E principal components, stability scores) were engineered as inputs alongside environmental covariates. Models were trained on 3,350 observations (1989-2014) from the USDA Uniform Soybean Trials (SoyURT) and tested on 276 held-out observations (2015-2019).

**Results:** The hybrid framework achieved R² = 0.71 and RMSE = 6.90 bu/ac. SHAP analysis confirmed that classical breeding features (BLUP, G×E PCs) were among the top predictive variables, validating the biology-guided AI approach. Genotype stability analysis identified both broadly adapted genotypes (e.g., ia2102) and environment-responsive genotypes (e.g., charleston) suitable for different breeding strategies.

**Conclusion:** Integrating classical quantitative genetics with machine learning creates interpretable, high-performing prediction systems. We provide an open-source implementation and interactive web dashboard for breeders.

**Keywords:** Genotype × Environment Interaction, BLUP, AMMI, XGBoost, Soybean, Climate Stress, Plant Breeding

---

## 1. Introduction

Climate change poses unprecedented challenges to global food security. Increasing temperature variability, altered precipitation patterns, and more frequent extreme weather events demand crop varieties that perform reliably across diverse and unpredictable environments (Ray et al., 2015). Plant breeders face the fundamental challenge of genotype × environment (G×E) interaction: the same genotype can perform differently across environments, complicating selection decisions.

Classical statistical approaches have long formed the backbone of G×E analysis. Best Linear Unbiased Prediction (BLUP) extracts genetic breeding values from mixed linear models (Henderson, 1975), while Additive Main Effects and Multiplicative Interaction (AMMI) and GGE biplot methods decompose the G×E interaction matrix to identify stable and high-performing genotypes (Gauch, 2006; Yan & Tinker, 2006). These methods are interpretable, statistically rigorous, and widely trusted by breeders.

Concurrently, machine learning (ML) has emerged as a powerful tool for genomic prediction and yield forecasting. Algorithms like Random Forest and XGBoost can capture complex non-linear relationships between environmental variables and crop performance (van Klompenburg et al., 2020). However, purely data-driven ML approaches often ignore decades of quantitative genetics knowledge, treating genotypes as categorical variables rather than leveraging biologically meaningful genetic parameters.

**Research Gap:** No existing framework systematically integrates classical breeding statistics as engineered features for machine learning models. This represents a missed opportunity: BLUPs and G×E principal components encode biological information that could guide ML models toward more interpretable and generalizable predictions.

**Our Contribution:**
1. A hybrid framework where BLUP and G×E decomposition outputs serve as input features to XGBoost
2. Demonstration on a large-scale public soybean dataset (39,006 observations)
3. SHAP analysis proving classical features drive model predictions
4. An open-source implementation with interactive web dashboard

---

## 2. Materials and Methods

### 2.1 Dataset

We used the USDA SoyURT (Uniform Regional Soybean Tests) dataset, comprising 39,006 yield observations from 4,257 experimental genotypes tested across 63 locations over 31 years (1989-2019) (Krause et al., 2022). The dataset is publicly available under CC BY 4.0 license.

The response variable was grain yield in bushels per acre (bu/ac). Environmental factors included latitude, longitude, altitude, year, and maturity group.

### 2.2 Classical Statistical Models

**BLUP Model:** We fitted a linear mixed model using the lme4 package in R: