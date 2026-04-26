"""
🌱 Agri-AI-G×E: Genotype × Environment Prediction Dashboard
------------------------------------------------------------
Hybrid AI system combining BLUP/AMMI classical genetics with XGBoost ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Agri-AI-G×E | Yield Predictor",
    page_icon="🌱",
    layout="wide"
)

# ============================================
# HEADER
# ============================================
st.title("🌱 AI-Driven Genotype × Environment Yield Predictor")
st.markdown("""
**Hybrid System:** Classical Breeding Statistics (BLUP/AMMI) + Machine Learning (XGBoost)  
*For climate-stressed crop breeding decisions*
""")
st.divider()

# ============================================
# LOAD DATA (CACHED)
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv('data/raw/soybean_pheno.csv')
    df['Environment'] = df['location'].astype(str) + '_' + df['year'].astype(str)
    
    # Load classical features
    classical = pd.read_csv('data/processed/classical_features.csv')
    
    # Merge
    df = df.merge(classical, left_on='G', right_on='GEN', how='left')
    df = df.dropna(subset=['BLUP', 'GxE_PC1', 'GxE_PC2', 'Stability'])
    
    # Create environmental aggregates
    env_means = df.groupby('Environment')['eBLUE'].mean().reset_index()
    env_means.columns = ['Environment', 'Env_Mean_Yield']
    df = df.merge(env_means, on='Environment', how='left')
    
    return df

@st.cache_resource
def train_model(df):
    feature_cols = ['Genotype_Mean_Yield', 'BLUP', 'GxE_PC1', 'GxE_PC2', 'Stability',
                    'Env_Mean_Yield', 'latitude', 'longitude', 'altitude', 'year']
    
    X = df[feature_cols].dropna()
    y = df.loc[X.index, 'eBLUE']
    
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    return model, feature_cols

df = load_data()
model, feature_cols = train_model(df)

# ============================================
# SIDEBAR: CONTROLS
# ============================================
st.sidebar.header("🎛️ Prediction Controls")

# Select Genotype
genotype_list = sorted(df['G'].unique())
selected_genotype = st.sidebar.selectbox("Select Genotype", genotype_list[:100])

# Get genotype-specific data
geno_data = df[df['G'] == selected_genotype].iloc[0]

st.sidebar.metric("Genotype BLUP (Breeding Value)", f"{geno_data['BLUP']:.2f}")
st.sidebar.metric("G×E Stability Score", f"{geno_data['Stability']:.3f}")

st.sidebar.divider()
st.sidebar.subheader("🌤️ Environmental Conditions")

# Environmental sliders
latitude = st.sidebar.slider("Latitude", 35.0, 48.0, float(geno_data['latitude']), 0.1)
longitude = st.sidebar.slider("Longitude", -105.0, -75.0, float(geno_data['longitude']), 0.1)
altitude = st.sidebar.slider("Altitude (m)", 0, 1500, int(geno_data['altitude']) if not pd.isna(geno_data['altitude']) else 300, 10)
year = st.sidebar.slider("Year", 2015, 2030, 2025)
env_mean_yield = st.sidebar.slider("Expected Environment Mean Yield (bu/ac)", 20.0, 100.0, float(geno_data['Env_Mean_Yield']) if not pd.isna(geno_data['Env_Mean_Yield']) else 55.0, 1.0)

# ============================================
# MAKE PREDICTION
# ============================================
input_data = pd.DataFrame({
    'Genotype_Mean_Yield': [geno_data['Genotype_Mean_Yield']],
    'BLUP': [geno_data['BLUP']],
    'GxE_PC1': [geno_data['GxE_PC1']],
    'GxE_PC2': [geno_data['GxE_PC2']],
    'Stability': [geno_data['Stability']],
    'Env_Mean_Yield': [env_mean_yield],
    'latitude': [latitude],
    'longitude': [longitude],
    'altitude': [altitude],
    'year': [year]
})

prediction = model.predict(input_data)[0]

# ============================================
# MAIN PANEL
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Yield Prediction")
    st.metric(
        label=f"Predicted Yield for **{selected_genotype}**",
        value=f"{prediction:.2f} bu/ac"
    )
    
    # Gauge-like visualization
    fig, ax = plt.subplots(figsize=(8, 2))
    overall_mean = df['eBLUE'].mean()
    ax.barh(['Predicted'], [prediction], color='darkgreen', height=0.5)
    ax.axvline(overall_mean, color='red', linestyle='--', linewidth=2, label=f'Overall Mean: {overall_mean:.1f} bu/ac')
    ax.set_xlabel('Yield (bu/ac)')
    ax.legend()
    ax.set_xlim(0, max(prediction * 1.5, 100))
    st.pyplot(fig)
    
    if prediction > overall_mean:
        st.success(f"✅ Above average (+{(prediction - overall_mean):.1f} bu/ac)")
    else:
        st.warning(f"⚠️ Below average ({(prediction - overall_mean):.1f} bu/ac)")

with col2:
    st.subheader("🧬 Genotype Profile")
    
    # Radar-like comparison
    fig, ax = plt.subplots(figsize=(6, 6))
    metrics = ['BLUP', 'Yield', 'Stability']
    values = [geno_data['BLUP'], geno_data['Genotype_Mean_Yield'], 
              max(0, 10 - geno_data['Stability']) * 5]  # Convert stability to 0-50 scale
    
    # Normalize for radar
    max_vals = [max(df['BLUP'].abs()), df['Genotype_Mean_Yield'].max(), 50]
    values_norm = [v/m if m > 0 else 0 for v, m in zip(values, max_vals)]
    values_norm.append(values_norm[0])  # Close the polygon
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    metrics_plot = metrics + [metrics[0]]
    
    ax = plt.subplot(111, polar=True)
    ax.fill(angles, values_norm, alpha=0.3, color='darkgreen')
    ax.plot(angles, values_norm, 'o-', color='darkgreen', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title(f'{selected_genotype} Profile', fontweight='bold')
    st.pyplot(fig)

# ============================================
# EXPLORE TAB
# ============================================
st.divider()
st.subheader("🔍 Explore the Dataset")

tab1, tab2 = st.tabs(["Top Genotypes", "G×E Visualization"])

with tab1:
    st.write("**Top 10 Genotypes by BLUP (Breeding Value)**")
    top_genos = df.groupby('G').agg({
        'BLUP': 'first',
        'Genotype_Mean_Yield': 'first',
        'Stability': 'first'
    }).sort_values('BLUP', ascending=False).head(10).reset_index()
    st.dataframe(top_genos.style.background_gradient(cmap='Greens', subset=['BLUP', 'Genotype_Mean_Yield']), width='stretch')

with tab2:
    st.write("**G×E Interaction: Yield Range by Environment**")
    top_locs = df['location'].value_counts().head(8).index
    subset = df[df['location'].isin(top_locs)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    subset.boxplot(column='eBLUE', by='location', ax=ax, patch_artist=True)
    ax.set_xlabel('Location')
    ax.set_ylabel('Yield (bu/ac)')
    ax.set_title('Yield Distribution Across Major Trial Locations')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown("""
---
**📄 Citation:** Zulqar Nain (2026). *An AI-Integrated Framework for Genotype × Environment Interaction Modeling and Yield Prediction in Climate-Stressed Crops*.  
**📂 GitHub:** [github.com/agristattools/agri-ai-gee](https://github.com/agristattools/agri-ai-gee)  
**📊 Data:** USDA SoyURT (CC BY 4.0) — Krause et al. (2022)
""")

st.success("🚀 Ready to share with professors! This app demonstrates hybrid AI-G×E in action.")