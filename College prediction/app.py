"""
Flask Web Server for the JoSAA IIT Prediction Engine.
Serves the frontend and exposes the prediction API.
"""

from flask import Flask, request, jsonify, send_from_directory
from engine import DataLoader, PredictionEngine
import os

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ── Initialize the engine once at startup ──
print("[INFO] Initializing data loader and prediction engine...")
loader = DataLoader()
engine = PredictionEngine(loader)
print("[INFO] Engine ready. Serving requests.")


@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory("static", "index.html")


@app.route("/api/metadata", methods=["GET"])
def metadata():
    """Return available filter options for the frontend dropdowns."""
    meta = loader.get_metadata()
    return jsonify(meta)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run prediction.
    Expects JSON body: { rank, category, gender }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        rank = int(data.get("rank", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Rank must be a valid integer"}), 400

    if rank <= 0:
        return jsonify({"error": "Rank must be a positive integer"}), 400

    category = data.get("category", "").strip()
    gender = data.get("gender", "").strip()

    if not category or not gender:
        return jsonify({"error": "All fields (rank, category, gender) are required"}), 400

    results = engine.predict(rank, category, gender)

    # Summary stats
    safe_count = sum(1 for r in results if r["risk_tier_code"] == 1)
    moderate_count = sum(1 for r in results if r["risk_tier_code"] == 2)
    ambitious_count = sum(1 for r in results if r["risk_tier_code"] == 3)

    return jsonify({
        "predictions": results,
        "summary": {
            "total": len(results),
            "safe": safe_count,
            "moderate": moderate_count,
            "ambitious": ambitious_count,
        },
        "query": {
            "rank": rank,
            "category": category,
            "gender": gender,
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
