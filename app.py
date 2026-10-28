from flask import Flask, render_template, request
import os
import cv2
import pytesseract
import pdfplumber
import joblib
import sqlite3
from datetime import datetime
from PIL import Image, ImageChops
from flask import send_from_directory
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"
app = Flask(__name__)

# ----------------- UPLOAD FOLDER -----------------
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# -------------------------------------------------

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            module TEXT,
            fraud_score INTEGER,
            verdict TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------- LOAD ML MODEL ----------------
model = joblib.load("fraud_model.pkl")

# ---------------- OCR FUNCTION ----------------
def extract_text(path):
    if path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    else:
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray)

# ---------------- TAMPER DETECTION ----------------
def detect_tampering(path):

    if path.endswith(".pdf"):
        return False

    original = Image.open(path)

    # ✅ FIX: Convert RGBA → RGB
    if original.mode == "RGBA":
        original = original.convert("RGB")

    temp_path = "temp.jpg"
    original.save(temp_path, "JPEG", quality=85)

    resaved = Image.open(temp_path)

    diff = ImageChops.difference(original, resaved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    os.remove(temp_path)

    return max_diff > 50
# ---------------- ML PREDICTION ----------------
def predict_fraud(text, tamper_flag):

    suspicious_words = ["edited", "modified", "duplicate", "scan copy"]
    keyword_hits = sum(word in text.lower() for word in suspicious_words)

    # Model expects 3 features (adjust if needed)
    features = [[keyword_hits, int(tamper_flag), 0]]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    fraud_score = int(probability * 100)

    return prediction, fraud_score

# ---------------- MAIN ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":

        ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

        def allowed_file(filename):
            return "." in filename and \
                   filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

        file = request.files.get("file")
        module = request.form.get("module")

        if not file or file.filename == "":
            return "No file uploaded"

        if not allowed_file(file.filename):
            return "Invalid file type"

        # Save file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Extract text
        text = extract_text(filepath)

        # Detect tampering (only for images)
        tamper_flag = detect_tampering(filepath)

        # ML prediction
        prediction, fraud_score = predict_fraud(text, tamper_flag)

        if prediction == 1:
            verdict = "⚠️ High Risk – Fraud Pattern Detected"
        else:
            verdict = "✅ Document Classified as Genuine"

        confidence = fraud_score / 100

        # -------- SAVE TO DATABASE --------
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO verifications
            (filename, module, fraud_score, verdict, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            file.filename,
            module,
            fraud_score,
            verdict,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()
        # -----------------------------------

        result = {
            "score": fraud_score,
            "verdict": verdict,
            "file_path": f"/uploads/{file.filename}"
        }

    return render_template("index.html", result=result)
# ---------------- ADMIN ROUTE ----------------
@app.route("/admin")
def admin():

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM verifications ORDER BY id DESC")
    records = cursor.fetchall()

    conn.close()

    return render_template("admin.html", records=records)

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)