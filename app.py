import streamlit as st
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor

# Set up page configurations for an academic/engineering look
st.set_page_config(
    page_title="Dragonfly Membrane Predictor",
    page_icon="🔬",
    layout="centered"
)

# Title and Description
st.title("🔬 Dragonfly Membrane Frequency Predictor")
st.markdown("""
This local web application utilizes a trained **Gaussian Process (GP) Surrogate Model** to instantly predict structural natural frequencies based on your design configuration.
""")
st.write("---")

# =====================================================================
# LOAD SURROGATE MODEL PAYLOAD
# =====================================================================
@st.cache_resource
def load_surrogate_payload():
    """Cache the model payload so it only loads once into memory"""
    with open("gp_production_payload.pkl", "rb") as f:
        payload = pickle.load(f)
    return payload

try:
    payload = load_surrogate_payload()
    scaler = payload['scaler_X']
    gp_models_dict = payload['gp_models']
    st.sidebar.success("✅ GP Surrogate Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading 'gp_production_payload.pkl': {e}")
    st.stop()

# =====================================================================
# SIDEBAR / INPUT CONFIGURATIONS
# =====================================================================
st.sidebar.header("Design Input Parameters")

# Sliders configured with reasonable engineering bounds based on your LHS design
thickness = st.sidebar.slider("Thickness (mm)", min_value=0.001, max_value=0.050, value=0.015, step=0.001, format="%.3f")
youngs_modulus = st.sidebar.slider("Young's Modulus (MPa)", min_value=1000.0, max_value=5000.0, value=2500.0, step=50.0)
density = st.sidebar.slider("Density (t/mm³)", min_value=1.0e-9, max_value=5.0e-9, value=1.5e-9, step=1.0e-10, format="%.2e")
poissons_ratio = st.sidebar.slider("Poisson's Ratio", min_value=0.20, max_value=0.45, value=0.33, step=0.01)

# =====================================================================
# PREDICTION PIPELINE
# =====================================================================
st.subheader("📊 Real-Time Model Predictions")

# 1. Package the inputs into a 2D array matching the exact 4-feature order
raw_inputs = np.array([[thickness, youngs_modulus, density, poissons_ratio]])

# 2. Scale the raw inputs using your historical training scaler
scaled_inputs = scaler.transform(raw_inputs)

# 3. Predict across all available frequency modes in your dictionary
st.markdown("### Calculated Natural Frequencies:")

cols = st.columns(len(gp_models_dict))

for idx, (mode_name, model) in enumerate(gp_models_dict.items()):
    # Predict the target mean value
    try:
        # Standard GP predict statement returning mean and standard deviation
        mean_pred, std_pred = model.predict(scaled_inputs, return_std=True)
        freq_val = mean_pred[0]
        uncertainty = std_pred[0]
    except Exception:
        # Fallback if your model payload handles standalone mean arrays
        freq_val = model.predict(scaled_inputs)[0]
        uncertainty = None

    # Handle nested array dimensions if present
    if isinstance(freq_val, np.ndarray):
        freq_val = freq_val[0]
    if isinstance(uncertainty, np.ndarray):
        uncertainty = uncertainty[0]

    # Render each frequency mode inside a separate clean visual metric block
    with cols[idx]:
        st.metric(
            label=f"🎵 {mode_name.upper().replace('_', ' ')}", 
            value=f"{freq_val:.2f} Hz"
        )
        if uncertainty is not None:
            st.caption(f"Uncertainty (±1σ): {uncertainty:.4f}")

st.write("---")
# Visual verification check box to display raw feature array
if st.checkbox("Show Raw Feature Input Vectors (Debugging)"):
    st.write("**Raw Features Matrix passed to Scaler:**", raw_inputs)
    st.write("**Standardized Vector passed to GP Regressor:**", scaled_inputs)