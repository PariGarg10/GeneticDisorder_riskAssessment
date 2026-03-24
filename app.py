from flask import Flask, render_template, request, send_file
import pickle
import pandas as pd
from ai_explainer import generate_precautions
import markdown
import os


app = Flask(__name__)

# Load trained model
pipeline = pickle.load(open("cardio_pipeline.pkl", "rb"))
BACKDROP_IMAGE_PATH = r"C:\Users\Pari Garg\.cursor\projects\c-Users-Pari-Garg-OneDrive-Desktop-Softwareproj-GeneticDisorder-riskAssessment\assets\c__Users_Pari_Garg_AppData_Roaming_Cursor_User_workspaceStorage_2cf2ff172a81efcc7da5951990aa2476_images_image-a3bcc788-b79f-496c-aabd-e76902e6663b.png"
BACKDROP_VIDEO_PATH = r"C:\Users\Pari Garg\OneDrive\Desktop\Softwareproj\GeneticDisorder_riskAssessment\assets\backdrop.mp4"

@app.route("/")
def home():
    return render_template(
        "home.html",
        has_backdrop_video=os.path.exists(BACKDROP_VIDEO_PATH),
    )

@app.route("/backdrop-image")
def backdrop_image():
    if os.path.exists(BACKDROP_IMAGE_PATH):
        return send_file(BACKDROP_IMAGE_PATH)
    return ("Backdrop image not found", 404)


@app.route("/backdrop-video")
def backdrop_video():
    if os.path.exists(BACKDROP_VIDEO_PATH):
        return send_file(BACKDROP_VIDEO_PATH, mimetype="video/mp4")
    return ("Backdrop video not found", 404)


@app.route("/assessment")
def assessment():
    return render_template("assessment.html")


@app.route("/start-assessment")
def start_assessment():
    return render_template("start_assessment.html")


@app.route("/about-us")
def about_us():
    return render_template("about_us.html")

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/get-started")
def get_started():
    return render_template("get_started.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    gender = int(request.form["gender"])
    ap_hi = int(request.form["ap_hi"])
    ap_lo = int(request.form["ap_lo"])
    cholesterol = int(request.form["cholesterol"])
    gluc = int(request.form["gluc"])
    smoke = int(request.form["smoke"])
    alco = int(request.form["alco"])
    active = int(request.form["active"])
    bmi = float(request.form["bmi"])
    relatives = int(request.form["relatives"])

    patient_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "bmi": bmi
    }])

    # -------------------------
    # ML Prediction
    # -------------------------
    base_prob = pipeline.predict_proba(patient_df)[0][1]

    # -------------------------
    # Genetic Multiplier
    # -------------------------
    if relatives == 0:
        RR = 1.0
    elif relatives == 1:
        RR = 1.6
    else:
        RR = 2.3

    adjusted_risk = 1 - (1 - base_prob) ** RR
    adjusted_risk = min(adjusted_risk, 0.95)

    risk_percent = round(adjusted_risk * 100, 2)

    # -------------------------
    # Risk Category
    # -------------------------
    if adjusted_risk < 0.25:
        level = "Low Risk"
    elif adjusted_risk < 0.50:
        level = "Borderline Risk"
    elif adjusted_risk < 0.75:
        level = "Elevated Risk"
    else:
        level = "High Risk"

    # -------------------------
    # Feature Importance
    # -------------------------
    model = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["scaler"].feature_names_in_
    importances = model.feature_importances_

    importance_df = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    top_features = ", ".join([f[0] for f in importance_df[:3]])
    genetic_flag = "Yes" if relatives > 0 else "No"

    # -------------------------
    # AI Explanation
    # -------------------------
    ai_text = generate_precautions(level, top_features, genetic_flag)
    ai_text = markdown.markdown(ai_text)



    return render_template(
        "result.html",
        prediction=risk_percent,
        ai_text=ai_text
    )

if __name__ == "__main__":
    app.run(debug=True)
