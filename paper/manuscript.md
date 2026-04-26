# A Hybrid Classical-Machine Learning Framework for Genotype × Environment Interaction Analysis and Yield Prediction in Soybean

**Zulqarnain***

*Independent Researcher*

*GitHub: [github.com/agristattools](https://github.com/agristattools)*

*Live Dashboard: [agri-ai-gee-mlmodel.streamlit.app](https://agri-ai-gee-mlmodel.streamlit.app)*

*Project Repository: [github.com/agristattools/agri-ai-gee](https://github.com/agristattools/agri-ai-gee)*

---

## Abstract

**Background:** Climate change increasingly threatens crop production, making accurate prediction of genotype performance across environments essential for plant breeding programs. While classical quantitative genetics (BLUP, AMMI) excels at estimating breeding values, and machine learning captures non-linear environmental responses, no established framework systematically integrates both approaches for genotype × environment (G×E) analysis.

**Objective:** We developed and validated a hybrid framework that engineers classical genetic parameters as features for machine learning-based yield prediction.

**Methods:** Using the USDA SoyURT dataset (39,006 observations; 4,257 soybean genotypes; 63 locations; 1989–2019), we extracted Best Linear Unbiased Predictions (BLUPs), G×E principal components via singular value decomposition, and genotype stability scores. These classical features were combined with environmental covariates (latitude, longitude, altitude, year, maturity group) and used to train three regression models—Linear Regression, Random Forest, and XGBoost—under temporal validation (training: 1989–2014; testing: 2015–2019). Model interpretability was assessed using SHAP (SHapley Additive exPlanations) analysis.

**Results:** Environmental variance dominated yield variation (σ²_e = 149.32), with broad-sense heritability H² = 0.243. The G×E decomposition captured 56.2% of interaction variance in the first two principal components. All models achieved strong predictive performance (R² = 0.70–0.74; RMSE = 6.50–6.99 bu/ac). SHAP analysis confirmed that classical features—particularly BLUP and G×E_PC1—ranked among the most influential predictors, validating the biology-guided AI approach. Combined BLUP-stability analysis identified both broadly adapted genotypes (e.g., ia2102) and environment-responsive genotypes (e.g., charleston) suitable for contrasting breeding strategies.

**Conclusion:** The integration of classical quantitative genetics with machine learning creates interpretable, high-performing prediction systems for plant breeding. The framework is provided as an open-source implementation with an interactive web dashboard, enabling practical adoption by breeding programs addressing climate-stressed production environments.

**Keywords:** Genotype × Environment Interaction; BLUP; GGE Biplot; XGBoost; SHAP; Soybean; Climate Adaptation; Plant Breeding; Machine Learning; Multi-Environment Trials

---

## 1. Introduction

Climate change has emerged as the preeminent challenge to global food security in the twenty-first century. Rising temperatures, altered precipitation regimes, and increasing frequencies of extreme weather events are disrupting agricultural production systems worldwide (Ray et al., 2015; Lobell et al., 2011). For staple crops including soybean (*Glycine max* L. Merr.)—a critical source of protein and oil for human consumption and animal feed—climate volatility translates directly into yield instability across growing regions (Mourtzinis et al., 2019). Projections suggest that without adaptive interventions, soybean yields could decline by 3–5% per degree Celsius of warming in major production zones (Zhao et al., 2017).

Plant breeding occupies the front line of climate adaptation for agriculture. Breeding programs aim to develop varieties that perform reliably under the specific environmental conditions in which they will be grown. However, the fundamental biological phenomenon of genotype × environment (G×E) interaction complicates this objective: the relative performance of genotypes changes across environments, meaning that a variety that excels in one location or year may underperform in another (Malosetti et al., 2013). Understanding, quantifying, and predicting G×E interaction is therefore essential for making informed breeding decisions in a changing climate.

Classical statistical methods have provided the analytical foundation for G×E analysis over several decades. Best Linear Unbiased Prediction (BLUP), derived from linear mixed models, separates genetic signal from environmental noise to produce reliable estimates of genotype breeding values (Henderson, 1975; Piepho et al., 2008). The Additive Main Effects and Multiplicative Interaction (AMMI) model and its GGE biplot variant decompose the G×E interaction matrix into principal components, enabling visualization of genotype stability and environmental relationships (Gauch, 2006; Yan & Tinker, 2006). These methods are interpretable, statistically rigorous, and deeply embedded in plant breeding practice. Their outputs—BLUPs, interaction principal components, and stability scores—carry rich biological information about genotype performance and environmental responsiveness.

Concurrently, machine learning (ML) has transformed predictive modeling across scientific disciplines, including agriculture. Algorithms such as Random Forest (Breiman, 2001) and gradient-boosted trees (XGBoost; Chen & Guestrin, 2016) can capture complex non-linear relationships between environmental variables and crop phenotypes, often outperforming linear models when sufficient training data are available (van Klompenburg et al., 2020). Deep learning architectures have shown promise for integrating high-dimensional genomic and environmental data into unified prediction frameworks (Montesinos-López et al., 2021; Washburn et al., 2021). However, purely data-driven ML approaches frequently treat genotypes as categorical identifiers, discarding the quantitative genetic information that decades of classical breeding research have shown to be predictive of performance.

A conspicuous gap exists in the current literature: no established framework systematically integrates classical G×E statistics as engineered features for machine learning models. This represents a missed opportunity for synergy. BLUPs encode estimates of genetic merit; G×E principal components quantify environmental responsiveness; stability metrics capture phenotypic robustness. When these biologically meaningful parameters are provided as inputs, ML models can operate on scientifically grounded representations of genotype performance rather than learning genetic patterns de novo from raw categorical labels. We term this approach "biology-guided AI" for plant breeding.

This study presents an integrated framework that bridges classical quantitative genetics and modern machine learning for G×E analysis and yield prediction. Our specific objectives were fourfold: (1) to develop a reproducible pipeline that extracts BLUPs and G×E decomposition features from multi-environment trial data; (2) to evaluate whether these classical features enhance machine learning-based yield prediction when combined with environmental covariates; (3) to assess, via SHAP analysis, the relative contribution of classical versus environmental features to model predictions; and (4) to deploy the resulting system as an open-source, interactive decision-support tool accessible to breeders.

We demonstrate the framework using the publicly available USDA SoyURT dataset comprising 39,006 yield observations from 4,257 soybean genotypes tested across 63 locations over 31 years—one of the largest multi-environment trial datasets available for method development. All code, processed data, and a live interactive dashboard are provided to facilitate adoption and extension by the breeding community.

---

## 2. Materials and Methods

### 2.1 Dataset Description

This study utilized the USDA Northern Region Uniform Soybean Tests (SoyURT) dataset, a publicly available multi-environment trial collection spanning 31 years (1989–2019) and encompassing 39,006 individual yield observations (Krause et al., 2022). The dataset comprises 4,257 experimental soybean genotypes evaluated across 63 locations, generating 591 unique location-year environmental combinations. Grain yield, expressed in bushels per acre (bu/ac), served as the primary response variable. Ancillary environmental covariates included geographic coordinates (latitude and longitude), elevation (altitude in meters), and maturity group classification (Groups II and III). The dataset is distributed under the Creative Commons CC BY 4.0 license and is accessible via the SoyURT R package. All raw and processed data files used in this study are archived in the project repository at [github.com/agristattools/agri-ai-gee](https://github.com/agristattools/agri-ai-gee).

### 2.2 Classical Statistical Framework

**2.2.1 Mixed Linear Model Analysis**

Best Linear Unbiased Predictions (BLUPs) for genotype effects were obtained using a linear mixed model implemented with the lme4 package (Bates et al., 2015) in R version 4.5.3 (R Core Team, 2026). The model was specified as:

$$Y_{ij} = \mu + G_i + E_j + \varepsilon_{ij}$$

where $Y_{ij}$ represents the observed yield of genotype $i$ in environment $j$, $\mu$ is the overall mean, $G_i \sim N(0, \sigma^2_g)$ is the random effect of genotype $i$, $E_j \sim N(0, \sigma^2_e)$ is the random effect of environment $j$, and $\varepsilon_{ij} \sim N(0, \sigma^2_\varepsilon)$ is the residual error. Variance components ($\sigma^2_g$, $\sigma^2_e$, $\sigma^2_\varepsilon$) were estimated via restricted maximum likelihood (REML). Broad-sense heritability was calculated as:

$$H^2 = \frac{\sigma^2_g}{\sigma^2_g + \sigma^2_\varepsilon}$$

Genotype BLUPs, representing estimated breeding values, were extracted and stored as classical features for downstream machine learning integration.

**2.2.2 Genotype × Environment Interaction Decomposition**

To quantify G×E interaction structure, the genotype-by-environment mean yield matrix was constructed by aggregating observations to genotype-environment means. The matrix was mean-centered to remove additive main effects, and Singular Value Decomposition (SVD) was performed:

$$\mathbf{X}_{G\times E} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

where $\mathbf{U}$ contains genotype scores, $\mathbf{\Sigma}$ contains singular values, and $\mathbf{V}^T$ contains environment loadings. Principal component (PC) scores for genotypes on the first two interaction axes (G×E_PC1, G×E_PC2) were extracted as quantitative measures of G×E responsiveness. Genotype stability was quantified as the Euclidean distance from the biplot origin:

$$\text{Stability}_i = \sqrt{PC1_i^2 + PC2_i^2}$$

where lower values indicate broader adaptation (less G×E sensitivity) and higher values indicate stronger environment-specific responses. All G×E computations were performed using base R with the metan package (Olivoto & Lúcio, 2020) for auxiliary functions.

### 2.3 Machine Learning Framework

**2.3.1 Feature Engineering**

The feature set was designed to bridge classical quantitative genetics and machine learning through a hybrid engineering approach. Thirteen predictor variables were constructed, comprising five classical genetic features and eight environmental covariates:

*Classical Features:*
- Genotype mean yield (Genotype_Mean_Yield)
- Estimated breeding value from BLUP (BLUP)
- First G×E interaction principal component (GxE_PC1)
- Second G×E interaction principal component (GxE_PC2)
- Stability score (Stability)

*Environmental Features:*
- Environment mean yield (Env_Mean_Yield)
- Latitude (decimal degrees)
- Longitude (decimal degrees)
- Altitude (meters)
- Trial year
- Maturity group indicators (II, III)

All features were merged with individual trial observations, and rows with missing values for any feature were excluded, yielding a final modeling dataset of 3,626 complete observations.

**2.3.2 Model Training and Validation**

A temporal validation strategy was employed to simulate realistic breeding scenarios where future environments are unknown. Observations from 1989 to 2014 (n = 3,350) were assigned to the training set, while observations from 2015 to 2019 (n = 276) constituted the holdout test set. This design ensures that model evaluation reflects predictive performance on genuinely unseen years.

Three regression models of increasing complexity were compared:

1. **Linear Regression:** A multiple linear model serving as the interpretable baseline.
2. **Random Forest:** An ensemble of 100 decision trees (Breiman, 2001) with default hyperparameters, capturing non-linear relationships without extensive tuning.
3. **XGBoost:** An optimized gradient-boosted tree ensemble (Chen & Guestrin, 2016) configured with 200 estimators, maximum tree depth of 6, and a learning rate of 0.05 to balance predictive power and generalization.

All models were implemented in Python 3.14 using the scikit-learn library (Pedregosa et al., 2011) for Linear Regression and Random Forest, and the XGBoost library for gradient boosting.

**2.3.3 Evaluation Metrics**

Model performance was assessed using three complementary metrics:

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

where $y_i$ is the observed yield, $\hat{y}_i$ is the predicted yield, and $\bar{y}$ is the mean observed yield.

**2.3.4 Model Interpretability**

To understand feature contributions and validate the hybrid modeling approach, SHapley Additive exPlanations (SHAP) values were computed using the TreeExplainer algorithm (Lundberg & Lee, 2017) applied to the trained XGBoost model. SHAP values quantify each feature's marginal contribution to individual predictions, enabling both global feature importance ranking and visualization of effect directions.

### 2.4 Interactive Web Application

An interactive decision-support dashboard was developed using Streamlit (Streamlit Inc., 2026) to demonstrate practical deployment of the hybrid framework. The application enables users to select genotypes from the SoyURT panel, adjust environmental parameters (latitude, longitude, altitude, year, and expected environment mean yield) via intuitive sliders, and receive real-time yield predictions from the pre-trained XGBoost model. The dashboard also provides exploratory visualizations including genotype ranking tables, yield distribution boxplots by location, and a radar chart displaying individual genotype profiles across multiple breeding value dimensions. The application is deployed and publicly accessible at [agri-ai-gee-mlmodel.streamlit.app](https://agri-ai-gee-mlmodel.streamlit.app).

### 2.5 Code and Data Availability

The complete project is publicly accessible as a GitHub repository at [github.com/agristattools/agri-ai-gee](https://github.com/agristattools/agri-ai-gee) under the MIT License. The repository contains all R scripts for classical statistical analysis, Python modules for data preprocessing and machine learning, Jupyter notebooks documenting exploratory and predictive analyses, processed data files including genotype BLUPs and G×E features, all publication figures, and the Streamlit dashboard source code.

---

## 3. Results

### 3.1 Variance Component Analysis and Heritability

The linear mixed model partitioned yield variation into its genetic and environmental components (Table 1). Environmental variance (σ²_e = 149.32) dominated the total phenotypic variation, substantially exceeding both genetic variance (σ²_g = 9.42) and residual variance (σ²_ε = 29.37). Broad-sense heritability was estimated as H² = 0.243, indicating that approximately 24% of the observed phenotypic variation in grain yield is attributable to genotypic differences, while the remaining 76% arises from environmental factors and unexplained residual effects. This pronounced environmental influence provides strong quantitative justification for incorporating detailed environmental covariates into prediction models.

**Table 1.** Variance components and heritability estimates from the linear mixed model.

| Variance Component | Symbol | Estimate | Std. Dev. | Percentage |
|-------------------|--------|----------|-----------|------------|
| Genotype | σ²_g | 9.42 | 3.07 | 5.0% |
| Environment | σ²_e | 149.32 | 12.22 | 79.4% |
| Residual | σ²_ε | 29.37 | 5.42 | 15.6% |
| **Broad-sense heritability** | **H²** | **0.243** | — | — |

### 3.2 Genotype × Environment Interaction Structure

Singular value decomposition of the mean-centered G×E interaction matrix revealed substantial interaction structure. The first principal component (PC1) explained 33.1% of the total G×E interaction variance, while the second principal component (PC2) accounted for an additional 23.1%, yielding a cumulative variance explained of 56.2% for the first two axes.

The GGE biplot (Figure 1 in repository: `results/figures/ammi_biplot.png`) visualizes the simultaneous ordination of genotypes and environments. Genotypes positioned near the biplot origin, such as ia2102 (Stability = 0.0003) and u11_920017 (Stability = 0.0003), exhibited minimal interaction with environments and represent broadly adapted lines suitable for cultivation across diverse conditions. In contrast, genotypes with large PC1 scores, notably charleston (PC1 = 18.60) and e91031 (PC1 = 14.93), demonstrated strong environment-specific responses. These responsive genotypes are candidates for targeted deployment in optimal environments where their specific adaptations confer maximum yield advantage.

Environments separated along PC1 primarily according to yield potential, with high-yielding environments clustering on the positive PC1 axis and lower-yielding environments on the negative axis. This pattern suggests that PC1 captures a productivity gradient that differentiates genotypes based on their responsiveness to favorable conditions.

### 3.3 Yield Distribution and Environmental Variation

Substantial yield variation was observed across the 31-year study period. Mean annual yield ranged from approximately 35 to 75 bu/ac, with an overall mean of 53.3 bu/ac across all trials (see `results/figures/yield_distribution.png` and `results/figures/environments_overview.png`). Yield distributions by maturity group revealed comparable central tendencies between Maturity Groups II and III, though Group III exhibited slightly greater dispersion, suggesting more variable performance across environments.

The top 15 trial locations by number of observations contributed 8,426 observations collectively, with individual location counts ranging from 365 to 1,214 trials. These high-volume locations, concentrated in the Midwestern United States, formed the core environmental sampling for the modeling framework.

### 3.4 Predictive Model Performance

All three regression models achieved strong predictive performance on the temporally independent test set (2015–2019), demonstrating the value of the hybrid classical-ML feature engineering approach (Table 2). Linear Regression produced the highest coefficient of determination (R² = 0.743, RMSE = 6.50 bu/ac), followed by XGBoost (R² = 0.711, RMSE = 6.90 bu/ac) and Random Forest (R² = 0.703, RMSE = 6.99 bu/ac).

The strong performance of Linear Regression indicates that the engineered features—particularly the classical genetic parameters—capture meaningful linear relationships with yield. The marginal underperformance of tree-based methods relative to linear regression on this structured breeding dataset suggests that the dominant yield-determining processes in these well-characterized trials are adequately modeled by linear combinations of genotype merit and environmental productivity.

**Table 2.** Model performance comparison on the temporally independent test set (2015–2019, n = 276).

| Model | R² | RMSE (bu/ac) | MAE (bu/ac) |
|-------|-----|--------------|-------------|
| Linear Regression | 0.743 | 6.50 | 4.95 |
| Random Forest | 0.703 | 6.99 | 5.23 |
| XGBoost | 0.711 | 6.90 | 5.28 |

Scatter plots of predicted versus observed yield (`results/figures/predicted_vs_actual.png`) confirm that all three models produced well-calibrated predictions without systematic bias across the yield range. Residuals were approximately normally distributed with no evidence of heteroscedasticity, supporting the validity of the modeling assumptions.

### 3.5 Feature Importance and Model Interpretability

SHAP analysis of the XGBoost model revealed the relative contribution of each feature to yield predictions (`results/figures/feature_importance.png` and `results/figures/shap_beeswarm.png`). Environment mean yield (Env_Mean_Yield) emerged as the single most influential predictor, consistent with the large environmental variance component identified in the mixed model analysis. Among the classical genetic features, BLUP and GxE_PC1 ranked second and third in importance, respectively, confirming that genotype breeding values and interaction scores contribute meaningful predictive signal beyond what environmental covariates alone provide.

The direction of SHAP effects aligned with biological expectations: higher BLUP values (indicating superior genetic merit) were associated with positive yield predictions, while higher stability scores (indicating greater G×E sensitivity) showed variable effects depending on environmental context. Latitude and longitude exhibited non-linear effects consistent with regional adaptation patterns, with specific geographic zones associated with yield advantages for certain genotype groups.

### 3.6 Genotype Recommendations for Breeding

Integration of BLUP rankings with stability analysis enabled dual-purpose genotype recommendations (Table 3). For broad adaptation programs targeting consistent performance across diverse environments, genotypes ia2102, u11_920017, and ld02_4485 were identified as optimal candidates, combining favorable BLUP values with minimal G×E sensitivity (stability scores < 0.001). These lines are suitable for release across multiple environments without requiring specific environmental targeting.

For niche adaptation strategies, highly responsive genotypes such as charleston (BLUP = −1.58, PC1 = 18.60) and e91031 (BLUP = 2.37, PC1 = 14.93) were identified. These genotypes exhibit strong environment-specific yield advantages when deployed in their optimal conditions, despite lower stability scores. Breeding programs pursuing specific adaptation to high-yield environments would benefit from advancing such responsive lines.

**Table 3.** Top genotypes identified by combined BLUP-stability analysis for contrasting breeding objectives.

| Objective | Genotype | BLUP | Stability | Mean Yield (bu/ac) |
|-----------|----------|------|-----------|---------------------|
| **Broad Adaptation** | ia2102 | −0.21 | 0.0003 | 56.40 |
| | u11_920017 | 0.02 | 0.0003 | 64.84 |
| | ld02_4485 | 0.57 | 0.0004 | 64.31 |
| **Specific Adaptation** | charleston | −1.58 | 20.14 | 52.37 |
| | e91031 | 2.37 | 16.18 | 54.42 |
| | apex | −0.44 | 13.99 | 45.55 |

---

## 4. Discussion

### 4.1 The Value of Hybrid Classical-ML Frameworks in Plant Breeding

This study demonstrates that classical quantitative genetics and machine learning are not competing paradigms but complementary tools whose integration yields interpretable, high-performing prediction systems. By engineering BLUPs and G×E principal components as input features, we created a "biology-guided" AI that respects the statistical foundations of plant breeding while leveraging the flexibility of modern machine learning algorithms.

The strong predictive performance across all three models (R² = 0.70–0.74) validates the feature engineering strategy itself. When biologically meaningful features are properly constructed, even simple linear regression achieves performance comparable to more complex tree-based ensembles. This finding echoes recent work in genomic prediction showing that feature quality often outweighs model complexity (Montesinos-López et al., 2021). For breeding programs with limited computational infrastructure, our results suggest that a well-designed linear model incorporating classical genetic parameters may suffice for operational yield prediction.

The SHAP analysis provided quantitative validation of the hybrid approach: BLUP and G×E_PC1 ranked among the top three most influential features in the XGBoost model. This demonstrates that classical breeding statistics contribute unique predictive information not captured by environmental covariates alone. The presence of G×E principal components in the top features confirms that statistical decomposition of interaction variance produces biologically relevant signals that ML models can exploit.

### 4.2 Environmental Dominance and Implications for Climate-Stressed Breeding

The variance component analysis revealed that environment explains approximately 79% of total yield variation (σ²_e = 149.32), dwarfing the genetic contribution (σ²_g = 9.42). This environmental dominance, while consistent with previous multi-environment soybean trials (Krause et al., 2022; Xavier et al., 2018), carries urgent implications for breeding under climate change.

As temperature and precipitation patterns become increasingly volatile, the environmental variance component is likely to expand further, making genotype performance prediction simultaneously more difficult and more critical. Our framework addresses this challenge through two mechanisms: (1) the explicit incorporation of environmental covariates enables prediction under novel environmental conditions, and (2) the stability metric derived from G×E decomposition allows breeders to quantify and select for phenotypic robustness.

The interactive web dashboard, deployed at [agri-ai-gee-mlmodel.streamlit.app](https://agri-ai-gee-mlmodel.streamlit.app), operationalizes these capabilities for practitioners. A breeder evaluating a candidate genotype for a new target environment can adjust environmental parameters and receive an immediate yield prediction, informed by 31 years of historical trial data. This represents a practical step toward "climate-aware" breeding decisions.

### 4.3 Stable Versus Responsive Genotypes: A Portfolio Approach

Our dual identification of broadly adapted and environment-responsive genotypes supports a portfolio-based breeding strategy (Döring et al., 2015). Genotypes like ia2102 and u11_920017, with stability scores approaching zero, provide reliable performance across diverse environments—the "insurance" component of a breeding portfolio. Conversely, genotypes like charleston and e91031, with large G×E PC scores, offer substantial yield premiums when matched to their optimal environments—the "growth" component.

This portfolio perspective is particularly relevant for climate-stressed regions where environmental predictability is declining. A combination of stable genotypes (for food security in adverse years) and responsive genotypes (for productivity in favorable years) may represent an optimal risk-management strategy. Our framework enables systematic identification of both genotype classes from historical trial data.

### 4.4 Comparison with Prior Work

Previous efforts to integrate statistical genetics with machine learning have primarily focused on genomic prediction, where marker effects estimated by Bayesian methods serve as features for deep learning models (Abdollahi-Arpanahi et al., 2020; Sandhu et al., 2021). Our work extends this integration paradigm to the G×E domain, demonstrating that non-genomic classical parameters—BLUPs, interaction PC scores, and stability metrics—similarly enhance ML-based prediction.

The performance levels we observe (R² ≈ 0.71–0.74) are consistent with published yield prediction studies in soybean and other crops (van Klompenburg et al., 2020). However, direct quantitative comparison across studies is complicated by differences in datasets, validation schemes, and feature sets. Our contribution is not a marginal improvement in R² but rather a methodological framework for systematic integration of classical and ML approaches, accompanied by open-source implementation and interactive deployment.

### 4.5 Limitations and Future Directions

Several limitations warrant acknowledgment. First, the temporal validation split, while ecologically valid, resulted in a relatively small test set (n = 276). This constrains statistical power for model comparison, and the observed performance ordering (Linear > XGBoost > Random Forest) should not be overinterpreted as a definitive ranking of model classes for all breeding datasets.

Second, environmental characterization was limited to geographic coordinates and derived aggregates. Incorporation of high-resolution weather time-series (temperature, precipitation, solar radiation during critical growth stages) could substantially improve the environmental feature space and enable more granular climate-stress modeling. Recent advances in remote sensing and IoT-enabled field phenotyping offer promising data sources for this enhancement (Araus et al., 2018).

Third, our current framework operates at the phenotypic level without genomic information. Integration of molecular markers (SNPs) would enable genomic prediction capabilities, extending the framework's utility to untested genotype-environment combinations—the ultimate goal of predictive breeding (Crossa et al., 2017).

Finally, validation across multiple crop species and diverse agroecological zones is needed to establish generalizability. The structured nature of the SoyURT dataset, with its replicated designs and managed trials, may produce different feature importance patterns than less-controlled on-farm trials in smallholder systems.

### 4.6 Practical Recommendations for Breeding Programs

Based on our findings, we offer the following operational recommendations for breeding programs seeking to implement hybrid AI-G×E frameworks:

1. **Maintain classical statistical pipelines** alongside ML development. BLUPs and G×E decompositions produce valuable features that enhance ML performance.
2. **Prioritize environmental data collection.** Geographic coordinates and site characterization require minimal additional cost but substantially improve prediction accuracy.
3. **Use stability metrics for multi-objective selection.** Combining BLUP rankings with stability scores enables simultaneous improvement of yield potential and environmental robustness.
4. **Deploy interactive decision tools.** The Streamlit dashboard demonstrates that complex analytical pipelines can be made accessible to breeders without programming expertise, bridging the gap between computational research and field application.

---

## 5. Conclusion

This study presents a hybrid framework that integrates classical quantitative genetics with machine learning for genotype × environment interaction analysis and yield prediction in soybean. By engineering BLUPs, G×E principal components, and stability scores as features for predictive models, we demonstrate that biologically meaningful statistical parameters enhance the performance and interpretability of machine learning in plant breeding applications.

Three principal findings emerge from this work. First, environmental factors dominate yield variation (σ²_e = 149.32; H² = 0.243), underscoring both the challenge that climate change poses to crop production and the necessity of incorporating environmental covariates into prediction systems. Second, classical genetic features—particularly BLUPs and G×E interaction scores—rank among the most influential predictors in machine learning models, confirming that statistical genetics provides complementary information not captured by raw environmental data alone. Third, simultaneous consideration of genetic merit and stability enables systematic identification of both broadly adapted genotypes for food security and environment-responsive genotypes for targeted production.

The framework's practical value is operationalized through an interactive web dashboard at [agri-ai-gee-mlmodel.streamlit.app](https://agri-ai-gee-mlmodel.streamlit.app) that enables breeders to explore genotype performance under user-specified environmental scenarios. This deployment demonstrates that complex computational pipelines can be made accessible to practitioners without programming expertise, addressing the persistent gap between methodological development and field application in agricultural research.

Future extensions of this work should pursue three directions: integration of genomic markers to enable prediction for untested genotype-environment combinations, incorporation of high-resolution weather time-series to refine environmental characterization, and validation across diverse crop species and agroecological zones to establish generalizability.

The open-source implementation at [github.com/agristattools/agri-ai-gee](https://github.com/agristattools/agri-ai-gee), complete with documented code, processed data, and a live demonstration, provides a foundation for continued development by the broader breeding and computational biology communities. We contend that the future of predictive plant breeding lies not in choosing between classical statistics and artificial intelligence, but in their thoughtful, principled integration.

---

## Data and Code Availability

- **GitHub Repository:** [https://github.com/agristattools/agri-ai-gee](https://github.com/agristattools/agri-ai-gee)
- **Live Interactive Dashboard:** [https://agri-ai-gee-mlmodel.streamlit.app](https://agri-ai-gee-mlmodel.streamlit.app)
- **Dataset:** USDA SoyURT, accessible via the SoyURT R package (Krause et al., 2022), CC BY 4.0 license
- **Author GitHub:** [https://github.com/agristattools](https://github.com/agristattools)

---

## References

1. Abdollahi-Arpanahi, R., Gianola, D., & Peñagaricano, F. (2020). Deep learning versus parametric and ensemble methods for genomic prediction of complex phenotypes. *Genetics Selection Evolution*, 52(1), 12.
2. Araus, J.L., Kefauver, S.C., Zaman-Allah, M., Olsen, M.S., & Cairns, J.E. (2018). Translating high-throughput phenotyping into genetic gain. *Trends in Plant Science*, 23(5), 451–466.
3. Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. *Journal of Statistical Software*, 67(1), 1–48.
4. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
6. Crossa, J., Pérez-Rodríguez, P., Cuevas, J., Montesinos-López, O., Jarquín, D., de los Campos, G., ... & Varshney, R.K. (2017). Genomic selection in plant breeding: Methods, models, and perspectives. *Trends in Plant Science*, 22(11), 961–975.
7. Döring, T.F., Annicchiarico, P., Clarke, S., Haigh, Z., Jones, H.E., Pearce, H., ... & Wolfe, M.S. (2015). Comparative analysis of performance and stability among composite cross populations, variety mixtures and pure lines of winter wheat in organic and conventional cropping systems. *Field Crops Research*, 183, 235–245.
8. Gauch, H.G. (2006). Statistical analysis of yield trials by AMMI and GGE. *Crop Science*, 46(4), 1488–1500.
9. Henderson, C.R. (1975). Best linear unbiased estimation and prediction under a selection model. *Biometrics*, 31(2), 423–447.
10. Krause, M.D., Dias, K.O.G., Singh, A.K., & Beavis, W.D. (2022). Using large soybean historical data to study genotype by environment variation and identify mega-environments with the integration of genetic and non-genetic factors. *bioRxiv*, doi: 10.1101/2022.04.11.487885.
11. Lobell, D.B., Schlenker, W., & Costa-Roberts, J. (2011). Climate trends and global crop production since 1980. *Science*, 333(6042), 616–620.
12. Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.
13. Malosetti, M., Ribaut, J.M., & van Eeuwijk, F.A. (2013). The statistical analysis of multi-environment data: Modeling genotype-by-environment interaction and its genetic basis. *Frontiers in Physiology*, 4, 44.
14. Montesinos-López, O.A., Montesinos-López, A., Pérez-Rodríguez, P., Barrón-López, J.A., Martini, J.W., Fajardo-Flores, S.B., ... & Crossa, J. (2021). A review of deep learning applications for genomic selection. *BMC Genomics*, 22(1), 19.
15. Mourtzinis, S., Specht, J.E., & Conley, S.P. (2019). Defining optimal soybean seeding rates and associated risk across North America. *Agronomy Journal*, 111(2), 800–812.
16. Olivoto, T., & Lúcio, A.D.C. (2020). metan: An R package for multi-environment trial analysis. *Methods in Ecology and Evolution*, 11(6), 783–789.
17. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
18. Piepho, H.P., Möhring, J., Melchinger, A.E., & Büchse, A. (2008). BLUP for phenotypic selection in plant breeding and variety testing. *Euphytica*, 161(1), 209–228.
19. R Core Team. (2026). R: A language and environment for statistical computing. R Foundation for Statistical Computing, Vienna, Austria.
20. Ray, D.K., Gerber, J.S., MacDonald, G.K., & West, P.C. (2015). Climate variation explains a third of global crop yield variability. *Nature Communications*, 6, 5989.
21. Sandhu, K.S., Lozada, D.N., Zhang, Z., Pumphrey, M.O., & Carter, A.H. (2021). Deep learning for predicting complex traits in spring wheat breeding program. *Frontiers in Plant Science*, 11, 613325.
22. van Klompenburg, T., Kassahun, A., & Catal, C. (2020). Crop yield prediction using machine learning: A systematic literature review. *Computers and Electronics in Agriculture*, 177, 105709.
23. Washburn, J.D., Mejia-Guerra, M.K., Ramstein, G., Kremling, K.A., Valluru, R., Buckler, E.S., & Wang, H. (2021). Evolutionarily informed deep learning methods for predicting relative transcript abundance from DNA sequence. *Proceedings of the National Academy of Sciences*, 118(14), e2023554118.
24. Xavier, A., Jarquin, D., Howard, R., Ramasubramanian, V., Specht, J.E., Graef, G.L., ... & Rainey, K.M. (2018). Genome-wide analysis of grain yield stability and environmental interactions in a multiparental soybean population. *G3: Genes, Genomes, Genetics*, 8(2), 519–529.
25. Yan, W., & Tinker, N.A. (2006). Biplot analysis of multi-environment trial data: Principles and applications. *Canadian Journal of Plant Science*, 86(3), 623–645.
26. Zhao, C., Liu, B., Piao, S., Wang, X., Lobell, D.B., Huang, Y., ... & Asseng, S. (2017). Temperature increase reduces global yields of major crops in four independent estimates. *Proceedings of the National Academy of Sciences*, 114(35), 9326–9331.

---

*Manuscript prepared for journal submission. Corresponding code and data available at [github.com/agristattools/agri-ai-gee](https://github.com/agristattools/agri-ai-gee). Interactive dashboard at [agri-ai-gee-mlmodel.streamlit.app](https://agri-ai-gee-mlmodel.streamlit.app).*