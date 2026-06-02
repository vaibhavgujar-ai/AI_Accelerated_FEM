import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# =====================================================================
# 1. LOAD SURROGATE MODEL PAYLOAD
# =====================================================================
GP_MODEL_PATH = "gp_production_payload.pkl"

if not os.path.exists(GP_MODEL_PATH):
    raise FileNotFoundError(
        f"Could not locate '{GP_MODEL_PATH}' in the current working directory. "
        f"Please verify the file path before starting the Flask server."
    )

print("Loading surrogate model payload and validation scales...")
with open(GP_MODEL_PATH, "rb") as f:
    payload = pickle.load(f)

# Hardcoded directly to your dictionary parameters verified by our debugger engine
scaler = payload['scaler_X']
gp_models_dict = payload['gp_models']

# =====================================================================
# 2. EMBEDDED UI JINJA2 HTML TEMPLATE (Bootstrap 5 Layout Engine)
# =====================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dragonfly Membrane Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f4f6f9; font-family: system-ui, -apple-system, sans-serif; }
        .card { border: none; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-radius: 8px; }
        .metric-card { background-color: #ffffff; border-left: 5px solid #1f77b4; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
        .btn-calculate { background-color: #1f77b4; border: none; transition: background 0.2s ease; }
        .btn-calculate:hover { background-color: #155a8a; }
        .dashed-box { display: flex; align-items: center; justify-content: center; height: 300px; color: #6c757d; border: 2px dashed #dee2e6; border-radius: 8px; background-color: #ffffff; }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-xl-11">
                <div class="text-center mb-4">
                    <h2 class="fw-bold text-dark">🔬 Dragonfly Membrane Frequency Predictor</h2>
                    <p class="text-muted">Gaussian Process Surrogate Model deployment console with statistical uncertainty estimation</p>
                </div>
                <hr class="mb-5">
                
                <div class="row g-4">
                    <div class="col-md-5">
                        <div class="card p-4">
                            <h4 class="mb-4 text-dark fw-bold" style="font-size: 1.15rem;">Design Input Metrics</h4>
                            <form method="POST">
                                <div class="mb-3">
                                    <label class="form-label text-secondary small fw-bold">Thickness_mm (mm)</label>
                                    <input type="number" step="0.0001" min="0.0001" max="1.0" name="thickness" class="form-control" value="{{ inputs.thickness }}" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label text-secondary small fw-bold">YoungsModulus_MPa (MPa)</label>
                                    <input type="number" step="0.1" name="youngs_modulus" class="form-control" value="{{ inputs.youngs_modulus }}" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label text-secondary small fw-bold">Density_t_mm3 (t/mm³)</label>
                                    <input type="text" name="density" class="form-control" value="{{ inputs.density }}" required>
                                    <small class="text-muted d-block mt-1" style="font-size: 0.75rem;">Accepts scientific notation formatting (e.g., 1.5e-9)</small>
                                </div>
                                <div class="mb-4">
                                    <label class="form-label text-secondary small fw-bold">PoissonsRatio</label>
                                    <input type="number" step="0.001" min="0.0" max="0.5" name="poissons_ratio" class="form-control" value="{{ inputs.poissons_ratio }}" required>
                                </div>
                                <button type="submit" class="btn btn-primary btn-calculate w-100 py-2 fw-bold text-white">⚡ Calculate Predictions</button>
                            </form>
                        </div>
                    </div>

                    <div class="col-md-7">
                        <div class="card p-4 h-100 bg-transparent shadow-none border-0 pt-0">
                            <h4 class="mb-4 text-dark fw-bold" style="font-size: 1.15rem;">📊 Predicted Modal Frequencies (95% CI)</h4>
                            {% if predictions %}
                                <div class="row g-3">
                                    {% for mode_name, data in predictions.items() %}
                                        <div class="col-12">
                                            <div class="p-3 metric-card">
                                                <small class="text-uppercase text-muted fw-bold d-block mb-1" style="font-size: 0.75rem; letter-spacing: 0.5px;">{{ mode_name }}</small>
                                                <div class="d-flex justify-content-between align-items-center">
                                                    <span class="fs-2 fw-bold text-dark">{{ "%.2f"|format(data.val) }} Hz</span>
                                                    <span class="text-secondary text-end small lh-sm">
                                                        <span class="fw-bold text-dark">95% Confidence Interval Bounds:</span><br>
                                                        <span class="font-monospace text-primary fw-bold">[{{ "%.2f"|format(data.lower) }} Hz — {{ "%.2f"|format(data.upper) }} Hz]</span> <br>
                                                        <span class="text-muted d-inline-block mt-1" style="font-size: 0.8rem;">Uncertainty margin (±1.96σ): ±{{ "%.4f"|format(data.uncertainty) }} Hz</span>
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    {% endfor %}
                                </div>
                            {% else %}
                                <div class="dashed-box">
                                    <div class="text-center">
                                        <span class="d-block mb-2 fs-5">😴 No Active Evaluations</span>
                                        <small class="text-muted">Adjust geometric values and click the calculation pipeline button.</small>
                                    </div>
                                </div>
                            {% endif %}
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </div>
</body>
</html>
"""

# =====================================================================
# 3. APP ROUTES & EVALUATION CONTROLLERS
# =====================================================================
@app.route("/", methods=["GET", "POST"])
def home():
    # Set default values for initial page load rendering parameters
    current_inputs = {
        "thickness": 0.015,
        "youngs_modulus": 2500.0,
        "density": "1.5e-9",
        "poissons_ratio": 0.33
    }
    predictions = None

    if request.method == "POST":
        # Capture raw form responses from client context
        current_inputs["thickness"] = float(request.form.get("thickness"))
        current_inputs["youngs_modulus"] = float(request.form.get("youngs_modulus"))
        current_inputs["density"] = request.form.get("density").strip()
        current_inputs["poissons_ratio"] = float(request.form.get("poissons_ratio"))

        # Type conversion supporting scientific notation syntax strings
        try:
            density_float = float(current_inputs["density"])
        except ValueError:
            density_float = 1.5e-9  # Fallback error recovery boundary

        # Package data into a 2D matrix matching the strict 4-feature training format
        raw_vector = np.array([[
            current_inputs["thickness"],
            current_inputs["youngs_modulus"],
            density_float,
            current_inputs["poissons_ratio"]
        ]])

        # Run standardization through your saved scikit-learn scaler step
        scaled_vector = scaler.transform(raw_vector)

        # Iterate across the submodels to execute evaluation mean and sigma tracking
        predictions = {}
        for mode_name, model in gp_models_dict.items():
            # Query the Gaussian Process asking for standard deviation variance paths
            pred_mean, pred_std = model.predict(scaled_vector, return_std=True)
            
            val = pred_mean[0]
            sigma = pred_std[0]
            
            # Type correction if values are wrapped inside redundant array dimensions
            if isinstance(val, np.ndarray): val = val[0]
            if isinstance(sigma, np.ndarray): sigma = sigma[0]
            
            # Compute statistical 95% threshold margin intervals (1.96 * sigma)
            margin_of_error = 1.96 * sigma
            lower_bound = val - margin_of_error
            upper_bound = val + margin_of_error
            
            # Map processed attributes to target dictionary keys for Jinja rendering loop
            predictions[mode_name.replace('_', ' ')] = {
                "val": val,
                "uncertainty": margin_of_error,
                "lower": lower_bound,
                "upper": upper_bound
            }

    return render_template_string(HTML_TEMPLATE, inputs=current_inputs, predictions=predictions)


if __name__ == "__main__":
    print("\n--- [FLASK DEVELOPMENT SERVER ACTIVATED] ---")
    print("Point your browser address to: http://127.0.0.1:5000\n")
    # Run server locally on default development port 5000
    app.run(debug=True, port=5000)