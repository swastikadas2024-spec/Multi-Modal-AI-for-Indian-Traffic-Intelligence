import os
import json
import torch
import pickle
import csv
from datetime import datetime
from flask import Flask, request, jsonify
from preprocess import normalize_text

app = Flask(__name__)

# Model and label loading
MODEL_PATH = "models/phase3_model.pth"
LABELS_PATH = "outputs/text_run/labels.txt"
ALERTS_CSV = "data/alerts.csv"

model = None
label_mapping = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the trained model and label mappings."""
    global model, label_mapping
    
    try:
        if os.path.exists(MODEL_PATH):
            model = torch.load(MODEL_PATH, map_location=device)
            model.eval()
            print(f"✓ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠ Model not found at {MODEL_PATH}. Using placeholder.")
            model = None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model = None
    
    try:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'r') as f:
                label_mapping = json.load(f)
            print(f"✓ Labels loaded from {LABELS_PATH}")
        else:
            # Placeholder labels for demo
            label_mapping = {
                "0": "normal",
                "1": "congestion",
                "2": "accident",
                "3": "potholes"
            }
            print(f"⚠ Using placeholder label mapping: {label_mapping}")
    except Exception as e:
        print(f"✗ Error loading labels: {e}")
        label_mapping = {"0": "unknown"}

def predict(text):
    """
    Predict the label for given text.
    
    Args:
        text (str): Input text to classify
    
    Returns:
        dict: {label, confidence, timestamp}
    """
    # Normalize text
    normalized_text = normalize_text(text)
    
    # If model not loaded, return placeholder prediction
    if model is None:
        return {
            "label": "placeholder",
            "confidence": 0.55,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "normalized_text": normalized_text,
            "warning": "Model not yet trained - placeholder prediction"
        }
    
    try:
        # Tokenize and prepare input (placeholder logic)
        # In real implementation, you'd use your actual tokenizer and model architecture
        with torch.no_grad():
            # Placeholder: random prediction (replace with actual model inference)
            prediction = torch.randint(0, len(label_mapping), (1,)).item()
            confidence = torch.rand(1).item() * 0.4 + 0.6  # 0.6-1.0
        
        label = label_mapping.get(str(prediction), "unknown")
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        return {
            "label": label,
            "confidence": round(confidence, 3),
            "timestamp": timestamp,
            "normalized_text": normalized_text
        }
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return {
            "label": "error",
            "confidence": 0.0,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": str(e)
        }

def log_alert(text, label, confidence, timestamp):
    """Log alert to CSV file."""
    os.makedirs("data", exist_ok=True)
    
    file_exists = os.path.exists(ALERTS_CSV)
    
    try:
        with open(ALERTS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'text', 'label', 'confidence'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': timestamp,
                'text': text,
                'label': label,
                'confidence': confidence
            })
        print(f"✓ Alert logged: {label} ({confidence})")
    except Exception as e:
        print(f"✗ Error logging alert: {e}")

def trigger_notifications(text, label, confidence):
    """Trigger Slack/Email notifications if configured."""
    try:
        from alert_handler import send_slack_alert, send_email_alert
        
        message = f"🚨 ALERT: {label.upper()} detected with {confidence*100:.1f}% confidence\n\nText: {text}"
        
        slack_webhook = os.getenv("SLACK_WEBHOOK")
        if slack_webhook:
            send_slack_alert(message, slack_webhook)
        
        email_password = os.getenv("EMAIL_PASSWORD")
        if email_password:
            email_recipient = os.getenv("ALERT_EMAIL_RECIPIENT", "admin@traffic.local")
            send_email_alert(message, email_recipient)
    except Exception as e:
        print(f"⚠ Notification error: {e}")

@app.route('/predict', methods=['POST'])
def predict_route():
    """
    POST /predict - Make predictions on traffic complaint text.
    
    Request JSON:
        {"text": "Traffic jam at MG Road"}
    
    Response JSON:
        {
            "label": "congestion",
            "confidence": 0.92,
            "alert": true/false,
            "timestamp": "2026-03-31T10:30:00Z"
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "'text' cannot be empty"}), 400
        
        # Get prediction
        result = predict(text)
        
        # Check if alert threshold exceeded
        alert_triggered = result.get('confidence', 0) > 0.8
        
        if alert_triggered:
            log_alert(text, result['label'], result['confidence'], result['timestamp'])
            trigger_notifications(text, result['label'], result['confidence'])
        
        return jsonify({
            "label": result['label'],
            "confidence": result['confidence'],
            "alert": alert_triggered,
            "timestamp": result['timestamp']
        }), 200
    
    except Exception as e:
        print(f"✗ Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """
    GET /alerts - Retrieve recent alerts.
    
    Query params:
        limit (int, default=10): Number of recent alerts to return
    
    Response JSON:
        {
            "count": 5,
            "alerts": [
                {"timestamp": "...", "text": "...", "label": "...", "confidence": 0.92},
                ...
            ]
        }
    """
    try:
        limit = request.args.get('limit', default=10, type=int)
        
        if not os.path.exists(ALERTS_CSV):
            return jsonify({
                "count": 0,
                "alerts": [],
                "message": "No alerts logged yet"
            }), 200
        
        alerts = []
        with open(ALERTS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                alerts.append(row)
        
        # Return last 'limit' alerts
        recent_alerts = alerts[-limit:]
        recent_alerts.reverse()
        
        return jsonify({
            "count": len(recent_alerts),
            "total_alerts": len(alerts),
            "alerts": recent_alerts
        }), 200
    
    except Exception as e:
        print(f"✗ Error in /alerts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }), 200

if __name__ == '__main__':
    print("=" * 60)
    print("Traffic Complaint API - Starting...")
    print("=" * 60)
    
    load_model()
    
    print(f"✓ Flask app starting on http://localhost:5000")
    print(f"✓ Available endpoints:")
    print(f"  - POST /predict - Make predictions")
    print(f"  - GET /alerts - Retrieve recent alerts")
    print(f"  - GET /health - Health check")
    print("=" * 60)
    
    app.run(host='127.0.0.1', port=5000, debug=True)
