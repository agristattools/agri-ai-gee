# ============================================
# BLUP MODEL - Genotype × Environment Analysis
# ============================================

# Load libraries
library(lme4)

# Load data
df <- read.csv("data/raw/soybean_pheno.csv")
cat("✅ Data loaded:", nrow(df), "observations\n")

# Convert categorical variables to factors
df$G <- as.factor(df$G)
df$location <- as.factor(df$location)
df$year <- as.factor(df$year)

# Create Environment factor (location-year combination)
df$Environment <- as.factor(paste(df$location, df$year, sep = "_"))
cat("✅ Created", length(unique(df$Environment)), "unique environments\n")

# -----------------------------------------------
# MODEL 1: Basic BLUP (Genotype + Environment)
# -----------------------------------------------
cat("\n🔬 Fitting BLUP Model...\n")
blup_model <- lmer(eBLUE ~ (1|G) + (1|Environment), data = df)

# Display variance components
cat("\n📊 Variance Components:\n")
print(summary(blup_model))

# Extract BLUPs (breeding values) for genotypes
genotype_blups <- ranef(blup_model)$G
colnames(genotype_blups) <- "BLUP"
genotype_blups$Genotype <- rownames(genotype_blups)
rownames(genotype_blups) <- NULL

# Sort by highest BLUP (best genotypes)
genotype_blups <- genotype_blups[order(-genotype_blups$BLUP), ]

cat("\n🏆 Top 10 Genotypes by BLUP:\n")
print(head(genotype_blups, 10))

# -----------------------------------------------
# MODEL 2: BLUP with G×E Interaction (Simplified)
# Note: Full G×E interaction has too many levels, 
# so we extract variance components from Model 1
# and will use AMMI for full G×E decomposition
# -----------------------------------------------
cat("\n⚠️  Note: Full G×E interaction model not possible (2.5M+ levels).\n")
cat("    Using AMMI model for G×E decomposition instead.\n")

# Extract variance components from Model 1
vc <- as.data.frame(VarCorr(blup_model))
cat("\n📊 Variance Components from BLUP:\n")
print(vc)

# Calculate heritability (broad-sense)
genetic_var <- vc$vcov[vc$grp == "G"]
env_var <- vc$vcov[vc$grp == "Environment"]
residual_var <- vc$vcov[vc$grp == "Residual"]
total_var <- genetic_var + env_var + residual_var
h2 <- genetic_var / (genetic_var + residual_var)  # H2 = Vg / (Vg + Ve)

cat(paste0("\n🧬 Broad-sense Heritability (H²): ", round(h2, 3)))
cat(paste0("\n   Genetic Variance: ", round(genetic_var, 2)))
cat(paste0("\n   Environmental Variance: ", round(env_var, 2)))
cat(paste0("\n   Residual Variance: ", round(residual_var, 2)))

# -----------------------------------------------
# SAVE RESULTS
# -----------------------------------------------

# Save genotype BLUPs for ML feature engineering
write.csv(genotype_blups, "data/processed/genotype_blups.csv", row.names = FALSE)
cat("\n\n✅ Genotype BLUPs saved to data/processed/genotype_blups.csv")

# Save variance components
write.csv(vc, "data/processed/variance_components.csv", row.names = FALSE)
cat("\n✅ Variance components saved to data/processed/variance_components.csv")

cat("\n\n🎉 BLUP modeling complete! Moving to AMMI next.\n")