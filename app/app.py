"""
🌱 Agri-AI-G×E: Genotype × Environment Prediction Dashboard
------------------------------------------------------------
Hybrid AI system combining BLUP/AMMI classical genetics with XGBoost ML
FEATURE: Upload your own trial data for custom predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
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
*Upload your own trial data or explore the pre-loaded USDA SoyURT dataset*
""")
st.divider()

# ============================================
# LOAD DATA & TRAIN MODEL (CACHED)
# ============================================
@st.cache_data
def load_training_data():
    df = pd.read_csv('data/raw/soybean_pheno.csv')
    df['Environment'] = df['location'].astype(str) + '_' + df['year'].astype(str)
    classical = pd.read_csv('data/processed/classical_features.csv')
    df = df.merge(classical, left_on='G', right_on='GEN', how='left')
    df = df.dropna(subset=['BLUP', 'GxE_PC1', 'GxE_PC2', 'Stability'])
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

df_train = load_training_data()
model, feature_cols = train_model(df_train)

# ============================================
# SIDEBAR: MODE SELECTION
# ============================================
st.sidebar.header("🎛️ Select Mode")

mode = st.sidebar.radio(
    "Choose how to use the predictor:",
    ["📊 Explore SoyURT Data", "📁 Upload My Own Data"]
)

st.sidebar.divider()

# ============================================
# MODE 1: EXPLORE PRE-LOADED DATA
# ============================================
if mode == "📊 Explore SoyURT Data":
    
    st.sidebar.subheader("🎛️ Prediction Controls")
    
    genotype_list = sorted(df_train['G'].unique())
    selected_genotype = st.sidebar.selectbox("Select Genotype", genotype_list[:100])
    
    geno_data = df_train[df_train['G'] == selected_genotype].iloc[0]
    
    st.sidebar.metric("Genotype BLUP", f"{geno_data['BLUP']:.2f}")
    st.sidebar.metric("Stability Score", f"{geno_data['Stability']:.3f}")
    
    st.sidebar.divider()
    st.sidebar.subheader("🌤️ Environment Settings")
    
    latitude = st.sidebar.slider("Latitude", 35.0, 48.0, float(geno_data['latitude']), 0.1)
    longitude = st.sidebar.slider("Longitude", -105.0, -75.0, float(geno_data['longitude']), 0.1)
    altitude = st.sidebar.slider("Altitude (m)", 0, 1500, int(geno_data['altitude']) if not pd.isna(geno_data['altitude']) else 300, 10)
    year = st.sidebar.slider("Year", 2015, 2030, 2025)
    env_mean_yield = st.sidebar.slider("Expected Environment Mean Yield", 20.0, 100.0, float(geno_data['Env_Mean_Yield']) if not pd.isna(geno_data['Env_Mean_Yield']) else 55.0, 1.0)
    
    # Prediction
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
    
    # Display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Yield Prediction")
        st.metric(label=f"Predicted Yield for {selected_genotype}", value=f"{prediction:.2f} bu/ac")
        
        fig, ax = plt.subplots(figsize=(8, 2))
        overall_mean = df_train['eBLUE'].mean()
        ax.barh(['Predicted'], [prediction], color='darkgreen', height=0.5)
        ax.axvline(overall_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {overall_mean:.1f} bu/ac')
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
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
        metrics = ['BLUP', 'Mean Yield', 'Stability']
        values = [geno_data['BLUP'], geno_data['Genotype_Mean_Yield'], 
                  max(0, 10 - geno_data['Stability']) * 5]
        max_vals = [max(df_train['BLUP'].abs()), df_train['Genotype_Mean_Yield'].max(), 50]
        values_norm = [v/m if m > 0 else 0 for v, m in zip(values, max_vals)]
        values_norm.append(values_norm[0])
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        metrics_plot = metrics + [metrics[0]]
        ax.fill(angles, values_norm, alpha=0.3, color='darkgreen')
        ax.plot(angles, values_norm, 'o-', color='darkgreen', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title(f'{selected_genotype} Profile', fontweight='bold')
        st.pyplot(fig)
    
    # Tabs
    st.divider()
    st.subheader("🔍 Explore SoyURT Dataset")
    tab1, tab2 = st.tabs(["Top Genotypes", "G×E Visualization"])
    
    with tab1:
        top_genos = df_train.groupby('G').agg({
            'BLUP': 'first', 'Genotype_Mean_Yield': 'first', 'Stability': 'first'
        }).sort_values('BLUP', ascending=False).head(10).reset_index()
        st.dataframe(top_genos.style.background_gradient(cmap='Greens', subset=['BLUP', 'Genotype_Mean_Yield']), width='stretch')
    
    with tab2:
        top_locs = df_train['location'].value_counts().head(8).index
        subset = df_train[df_train['location'].isin(top_locs)]
        fig, ax = plt.subplots(figsize=(10, 6))
        subset.boxplot(column='eBLUE', by='location', ax=ax, patch_artist=True)
        ax.set_xlabel('Location')
        ax.set_ylabel('Yield (bu/ac)')
        ax.set_title('Yield Distribution Across Major Trial Locations')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)


# ============================================
# MODE 2: USER UPLOAD MODE
# ============================================
elif mode == "📁 Upload My Own Data":
    
    st.sidebar.subheader("📋 Instructions")
    st.sidebar.markdown("""
    **Required columns:**
    - `Genotype` (name/ID)
    - `latitude`
    - `longitude`
    
    **Optional columns:**
    - `altitude`
    - `year`
    - `yield` (if you have it, for comparison)
    
    Upload a CSV file below.
    """)
    
    st.sidebar.divider()
    
    uploaded_file = st.sidebar.file_uploader("Upload your trial data (CSV)", type="csv")
    
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        
        st.subheader("📋 Your Uploaded Data")
        st.dataframe(user_df.head(10), width='stretch')
        st.caption(f"Total rows: {user_df.shape[0]}")
        
        # Check required columns
        required_cols = ['Genotype', 'latitude', 'longitude']
        missing = [c for c in required_cols if c not in user_df.columns]
        
        if missing:
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
            st.info("Please ensure your CSV has at least: Genotype, latitude, longitude")
        else:
            st.success("✅ All required columns present!")
            
            # Add missing optional columns with defaults
            if 'altitude' not in user_df.columns:
                user_df['altitude'] = 300
                st.info("ℹ️ 'altitude' not found — using default: 300m")
            if 'year' not in user_df.columns:
                user_df['year'] = 2025
                st.info("ℹ️ 'year' not found — using default: 2025")
            
            # Since we can't compute BLUP/G×E for new genotypes without multi-env data,
            # we use population averages as reasonable approximations
            user_df['Genotype_Mean_Yield'] = df_train['Genotype_Mean_Yield'].mean()
            user_df['BLUP'] = 0  # Neutral BLUP for unknown genotypes
            user_df['GxE_PC1'] = 0
            user_df['GxE_PC2'] = 0
            user_df['Stability'] = df_train['Stability'].mean()
            user_df['Env_Mean_Yield'] = df_train['eBLUE'].mean()
            
            # Predict
            X_user = user_df[feature_cols]
            user_df['Predicted_Yield_bu_ac'] = model.predict(X_user)
            
            st.divider()
            st.subheader("🎯 Predictions for Your Genotypes")
            
            # Display results
            result_cols = ['Genotype', 'latitude', 'longitude', 'year', 'Predicted_Yield_bu_ac']
            if 'yield' in user_df.columns:
                result_cols.append('yield')
                # Calculate accuracy if actual yield provided
                user_df['Residual'] = user_df['yield'] - user_df['Predicted_Yield_bu_ac']
                result_cols.append('Residual')
            
            display_df = user_df[result_cols].round(2)
            st.dataframe(display_df, width='stretch')
            
            # Visualization
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Predicted Yield Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(user_df['Predicted_Yield_bu_ac'], bins=20, color='steelblue', edgecolor='white')
                ax.axvline(user_df['Predicted_Yield_bu_ac'].mean(), color='red', linestyle='--', 
                          label=f"Mean: {user_df['Predicted_Yield_bu_ac'].mean():.1f}")
                ax.set_xlabel('Predicted Yield (bu/ac)')
                ax.set_ylabel('Frequency')
                ax.legend()
                st.pyplot(fig)
            
            with col2:
                if 'yield' in user_df.columns:
                    st.subheader("📈 Predicted vs Actual")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(user_df['yield'], user_df['Predicted_Yield_bu_ac'], alpha=0.6, color='darkgreen')
                    lims = [min(user_df[['yield', 'Predicted_Yield_bu_ac']].min()), 
                            max(user_df[['yield', 'Predicted_Yield_bu_ac']].max())]
                    ax.plot(lims, lims, 'r--', linewidth=2)
                    ax.set_xlabel('Actual Yield (bu/ac)')
                    ax.set_ylabel('Predicted Yield (bu/ac)')
                    if len(user_df) > 1:
                        r2 = r2_score(user_df['yield'], user_df['Predicted_Yield_bu_ac'])
                        ax.set_title(f'R² = {r2:.3f}')
                    st.pyplot(fig)
                else:
                    st.subheader("🏆 Top Predicted Genotypes")
                    top_pred = user_df.nlargest(10, 'Predicted_Yield_bu_ac')[['Genotype', 'Predicted_Yield_bu_ac']]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.barh(top_pred['Genotype'], top_pred['Predicted_Yield_bu_ac'], color='darkgreen')
                    ax.set_xlabel('Predicted Yield (bu/ac)')
                    ax.set_title('Top 10 Predicted Genotypes')
                    ax.invert_yaxis()
                    st.pyplot(fig)
            
            # Download button
            csv = user_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv,
                file_name='predicted_yields.csv',
                mime='text/csv'
            )
            
            st.success("🚀 Predictions complete! Download your results above.")
    
    else:
        st.info("👈 Upload your CSV file in the sidebar to get started!")
        st.markdown("""
        ### Example CSV Format:
        ```csv
        Genotype,latitude,longitude,altitude,year
        G1,40.1,-88.2,230,2025
        G2,42.3,-93.7,280,2025
        G3,38.9,-95.2,310,2025
                    """)

============================================
FOOTER
============================================
st.divider()
st.markdown("""

📄 Citation: Zulqarnain (2026). An AI-Integrated Framework for Genotype × Environment Interaction Modeling and Yield Prediction in Climate-Stressed Crops.
📂 GitHub: github.com/agristattools/agri-ai-gee
📊 Training Data: USDA SoyURT (CC BY 4.0) — Krause et al. (2022)
⚠️ Note: Predictions for user-uploaded genotypes use population-average genetic parameters. For optimal accuracy, provide multi-environment trial data.
""")