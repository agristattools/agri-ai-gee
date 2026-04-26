# ============================================
# G×E INTERACTION ANALYSIS (Robust Approach)
# Uses GGE model + stability analysis
# ============================================

library(metan)
library(ggplot2)

# Load data
df <- read.csv("data/raw/soybean_pheno.csv")
df$Environment <- as.factor(paste(df$location, df$year, sep = "_"))

cat("✅ Data loaded:", nrow(df), "observations\n")

# -----------------------------------------------
# AGGREGATE: Mean yield per Genotype × Environment
# This removes the need for reps in AMMI
# -----------------------------------------------
df_means <- aggregate(eBLUE ~ G + Environment, data = df, FUN = mean, na.rm = TRUE)
cat("✅ Aggregated to means:", nrow(df_means), "G×E combinations\n")

# -----------------------------------------------
# SUBSET: Take genotypes and environments with most data
# to create a balanced-ish matrix for AMMI
# -----------------------------------------------
top_geno <- names(sort(table(df_means$G), decreasing = TRUE))[1:60]
top_env <- names(sort(table(df_means$Environment), decreasing = TRUE))[1:20]

df_ammi <- df_means[df_means$G %in% top_geno & df_means$Environment %in% top_env, ]
df_ammi$G <- droplevels(as.factor(df_ammi$G))
df_ammi$Environment <- droplevels(as.factor(df_ammi$Environment))

cat("\n📊 AMMI Working Set:\n")
cat("   Genotypes:", length(unique(df_ammi$G)), "\n")
cat("   Environments:", length(unique(df_ammi$Environment)), "\n")
cat("   Observations:", nrow(df_ammi), "\n")

# Check minimum observations per G and E
cat("   Min obs per genotype:", min(table(df_ammi$G)), "\n")
cat("   Min obs per environment:", min(table(df_ammi$Environment)), "\n")

# -----------------------------------------------
# WIDE FORMAT MATRIX (G × E matrix)
# -----------------------------------------------
library(tidyr)
ge_matrix <- df_ammi %>%
  pivot_wider(
    id_cols = G,
    names_from = Environment,
    values_from = eBLUE,
    values_fn = mean  # Handle any duplicates
  )

# Convert to matrix
geno_names <- ge_matrix$G
ge_mat <- as.matrix(ge_matrix[, -1])
rownames(ge_mat) <- geno_names

# Check for missing values
n_missing <- sum(is.na(ge_mat))
cat("\n   Missing values in G×E matrix:", n_missing, "out of", length(ge_mat), "\n")

# -----------------------------------------------
# GGE BIPLOT via SVD (works on G×E matrix)
# -----------------------------------------------
cat("\n🔬 Performing SVD for G×E decomposition...\n")

# Mean-center the matrix (remove additive main effects)
ge_mat_centered <- ge_mat
for(i in 1:nrow(ge_mat_centered)) {
  ge_mat_centered[i, ] <- ge_mat_centered[i, ] - mean(ge_mat_centered[i, ], na.rm = TRUE)
}
for(j in 1:ncol(ge_mat_centered)) {
  ge_mat_centered[, j] <- ge_mat_centered[, j] - mean(ge_mat_centered[, j], na.rm = TRUE)
}

# Replace NAs with 0 for SVD (after centering)
ge_mat_centered[is.na(ge_mat_centered)] <- 0

# Singular Value Decomposition
svd_result <- svd(ge_mat_centered)

# Variance explained
var_explained <- round((svd_result$d^2 / sum(svd_result$d^2)) * 100, 2)
cat("   PC1 explains:", var_explained[1], "% of G×E variance\n")
cat("   PC2 explains:", var_explained[2], "% of G×E variance\n")
cat("   Cumulative (PC1+PC2):", var_explained[1] + var_explained[2], "%\n")

# Genotype PC scores
genotype_pc <- data.frame(
  GEN = geno_names,
  PC1 = svd_result$u[, 1] * svd_result$d[1],
  PC2 = svd_result$u[, 2] * svd_result$d[2],
  Mean_Yield = rowMeans(ge_mat, na.rm = TRUE)
)
rownames(genotype_pc) <- NULL

# Environment PC scores
env_names <- colnames(ge_mat)
environment_pc <- data.frame(
  ENV = env_names,
  PC1 = svd_result$v[, 1] * svd_result$d[1],
  PC2 = svd_result$v[, 2] * svd_result$d[2],
  Mean_Yield = colMeans(ge_mat, na.rm = TRUE)
)

cat("\n🏆 Top 10 genotypes by PC1 (strongest G×E interaction):\n")
top_pc1 <- genotype_pc[order(-abs(genotype_pc$PC1)), ]
print(head(top_pc1[, c("GEN", "PC1", "PC2", "Mean_Yield")], 10))

# -----------------------------------------------
# GGE BIPLOT (Publication Figure)
# -----------------------------------------------
cat("\n📈 Generating GGE biplot...\n")

png("results/figures/ammi_biplot.png", width = 10, height = 8, units = "in", res = 300)

plot(genotype_pc$PC1, genotype_pc$PC2, 
     type = "n",
     xlab = paste0("PC1 (", var_explained[1], "%)"),
     ylab = paste0("PC2 (", var_explained[2], "%)"),
     main = "GGE Biplot: Genotype × Environment Interaction",
     xlim = range(c(genotype_pc$PC1, environment_pc$PC1)) * 1.2,
     ylim = range(c(genotype_pc$PC2, environment_pc$PC2)) * 1.2)

# Add reference lines
abline(h = 0, lty = 2, col = "gray70")
abline(v = 0, lty = 2, col = "gray70")

# Plot genotypes in blue
points(genotype_pc$PC1, genotype_pc$PC2, col = "steelblue", pch = 1, cex = 0.8)
text(genotype_pc$PC1, genotype_pc$PC2, labels = genotype_pc$GEN, 
     col = "steelblue", cex = 0.5, pos = 3)

# Plot environments in red
points(environment_pc$PC1, environment_pc$PC2, col = "red", pch = 17, cex = 1.2)
text(environment_pc$PC1, environment_pc$PC2, labels = substr(environment_pc$ENV, 1, 12), 
     col = "red", cex = 0.6, pos = 3)

legend("topright", legend = c("Genotype", "Environment"), 
       col = c("steelblue", "red"), pch = c(1, 17), cex = 0.8)

dev.off()
cat("✅ Biplot saved to results/figures/ammi_biplot.png\n")

# -----------------------------------------------
# STABILITY ANALYSIS: Which genotypes are most stable?
# -----------------------------------------------
genotype_pc$Stability <- sqrt(genotype_pc$PC1^2 + genotype_pc$PC2^2)
genotype_pc$Stability_Rank <- rank(genotype_pc$Stability)

cat("\n🏆 Top 10 MOST STABLE genotypes (lowest PC distance from origin):\n")
stable <- genotype_pc[order(genotype_pc$Stability), ]
print(head(stable[, c("GEN", "Stability", "Mean_Yield")], 10))

cat("\n⚡ Top 10 MOST RESPONSIVE genotypes (highest PC distance):\n")
responsive <- genotype_pc[order(-genotype_pc$Stability), ]
print(head(responsive[, c("GEN", "Stability", "Mean_Yield")], 10))

# -----------------------------------------------
# SAVE FEATURES FOR ML PIPELINE
# -----------------------------------------------
blups <- read.csv("data/processed/genotype_blups.csv")
names(blups)[names(blups) == "Genotype"] <- "GEN"

classical_features <- merge(blups, genotype_pc[, c("GEN", "PC1", "PC2", "Stability", "Mean_Yield")], 
                            by = "GEN", all.x = TRUE)
names(classical_features)[names(classical_features) == "PC1"] <- "GxE_PC1"
names(classical_features)[names(classical_features) == "PC2"] <- "GxE_PC2"
names(classical_features)[names(classical_features) == "Mean_Yield"] <- "Genotype_Mean_Yield"

write.csv(classical_features, "data/processed/classical_features.csv", row.names = FALSE)
cat("\n✅ Classical features (BLUP + G×E PCs + Stability) saved to data/processed/classical_features.csv\n")

cat("\n🎉 G×E Analysis complete! Ready for ML pipeline.\n")