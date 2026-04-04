from flask import Flask, render_template, request, jsonify
import csv
from datetime import datetime
import os
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

app = Flask(__name__)
INCIDENTS_CSV = 'data/incidents.csv'
CONGESTION_CSV = 'data/congestion_predictions.csv'
VEHICLE_CSV = 'data/vehicle_detections.csv'

# loading the three models
with open('models/phase1_model.pkl', 'rb') as f:
    phase1_model = pickle.load(f)
with open('models/phase1_vectorizer.pkl', 'rb') as f:
    phase1_vectorizer = pickle.load(f)
with open('models/phase2_model.pkl', 'rb') as f:
    phase2_model = pickle.load(f)
with open('models/phase2_weather_encoder.pkl', 'rb') as f:
    le_weather = pickle.load(f)

model_cv = models.resnet18(pretrained=False)
model_cv.fc = nn.Linear(model_cv.fc.in_features, 2)
model_cv.load_state_dict(torch.load('models/phase3_model.pth',
                         map_location='cpu'))
model_cv.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def ensure_data_dir():
    os.makedirs('data', exist_ok=True)

def classify_urgency(category):
    urgency_map = {
        'congestion': 'High urgency',
        'signal_issue': 'Medium urgency',
        'road_damage': 'High urgency',
        'public_transport': 'Medium urgency',
        'road_safety': 'Critical urgency'
    }
    return urgency_map.get(category, 'Medium urgency')

def get_response_department(category):
    department_map = {
        'congestion': 'Traffic Police',
        'signal_issue': 'Traffic Engineering',
        'road_damage': 'Municipal Corporation',
        'public_transport': 'Transport Department',
        'road_safety': 'Traffic Police'
    }
    return department_map.get(category, 'Traffic Control Room')

def save_incident(record):
    ensure_data_dir()
    fieldnames = [
        'timestamp', 'language', 'complaint', 'category', 'urgency',
        'department', 'location', 'latitude', 'longitude', 'status', 'reporter'
    ]
    if os.path.exists(INCIDENTS_CSV):
        with open(INCIDENTS_CSV, 'r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []
        if 'latitude' not in existing_fields or 'longitude' not in existing_fields:
            with open(INCIDENTS_CSV, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({name: row.get(name, '') for name in fieldnames})
    with open(INCIDENTS_CSV, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow({name: record.get(name, '') for name in fieldnames})

def load_recent_incidents(limit=10):
    if not os.path.exists(INCIDENTS_CSV):
        return []
    with open(INCIDENTS_CSV, 'r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        incidents = list(reader)
    recent = incidents[-limit:]
    recent.reverse()
    return recent

def append_csv_row(csv_path, fieldnames, record):
    ensure_data_dir()
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists or handle.tell() == 0:
            writer.writeheader()
        writer.writerow({name: record.get(name, '') for name in fieldnames})

def load_recent_csv_rows(csv_path, limit=50):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    recent = rows[-limit:]
    recent.reverse()
    return recent

def urgency_to_score(urgency):
    score_map = {
        'Critical urgency': 100,
        'High urgency': 80,
        'Medium urgency': 55
    }
    return score_map.get(urgency, 50)

def congestion_to_score(level):
    score_map = {
        'HIGH': 90,
        'MEDIUM': 60,
        'LOW': 25
    }
    return score_map.get((level or '').upper(), 50)

def vehicle_to_score(result):
    return 75 if 'Truck' in (result or '') else 35

def level_from_score(score):
    if score >= 75:
        return 'HIGH'
    if score >= 45:
        return 'MEDIUM'
    return 'LOW'

def category_to_weight(category):
    weight_map = {
        'congestion': 100,
        'road_safety': 95,
        'road_damage': 80,
        'signal_issue': 65,
        'public_transport': 55
    }
    return weight_map.get(category, 50)

def get_live_traffic_factors():
    congestion_rows = load_recent_csv_rows(CONGESTION_CSV, 40)
    vehicle_rows = load_recent_csv_rows(VEHICLE_CSV, 40)

    if congestion_rows:
        congestion_scores = [congestion_to_score(row.get('congestion')) for row in congestion_rows]
        weather_pressure = sum(
            8 if (row.get('weather', '').lower() in ('rain', 'fog', 'snow')) else 0
            for row in congestion_rows
        )
        avg_congestion = sum(congestion_scores) / len(congestion_scores)
        congestion_factor = min(100, avg_congestion + weather_pressure / max(len(congestion_rows), 1))
    else:
        congestion_factor = 50

    if vehicle_rows:
        heavy_count = sum(1 for row in vehicle_rows if 'Truck' in row.get('result', ''))
        vehicle_factor = min(100, 30 + (heavy_count / len(vehicle_rows)) * 70)
    else:
        vehicle_factor = 30

    return {
        'congestion_factor': round(congestion_factor, 1),
        'vehicle_factor': round(vehicle_factor, 1)
    }

def traffic_score_for_incident(incident, live_factors):
    complaint_score = category_to_weight(incident.get('category', ''))
    urgency_score = urgency_to_score(incident.get('urgency', ''))
    congestion_factor = live_factors['congestion_factor']
    vehicle_factor = live_factors['vehicle_factor']

    score = (
        (complaint_score * 0.35) +
        (urgency_score * 0.30) +
        (congestion_factor * 0.20) +
        (vehicle_factor * 0.15)
    )
    score = max(0, min(100, round(score, 1)))
    return {
        'score': score,
        'level': level_from_score(score)
    }

# routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify_complaint', methods=['POST'])
def classify_complaint():
    complaint = request.json.get('complaint', '')
    vec = phase1_vectorizer.transform([complaint])
    category = phase1_model.predict(vec)[0]
    return jsonify({
        'category': category.replace('_', ' ').title(),
        'urgency':  classify_urgency(category)
    })

@app.route('/report_incident', methods=['POST'])
def report_incident():
    data = request.json or {}
    complaint = data.get('complaint', '').strip()
    if not complaint:
        return jsonify({'error': 'Complaint text is required'}), 400

    language = data.get('language', 'English').strip() or 'English'
    location = data.get('location', '').strip() or 'Unspecified'
    reporter = data.get('reporter', '').strip() or 'Anonymous'
    latitude = data.get('latitude', '')
    longitude = data.get('longitude', '')

    try:
        latitude = float(latitude) if latitude not in ('', None) else ''
    except (TypeError, ValueError):
        latitude = ''
    try:
        longitude = float(longitude) if longitude not in ('', None) else ''
    except (TypeError, ValueError):
        longitude = ''

    vec = phase1_vectorizer.transform([complaint])
    category = phase1_model.predict(vec)[0]
    urgency = classify_urgency(category)
    department = get_response_department(category)
    timestamp = datetime.utcnow().isoformat() + 'Z'

    save_incident({
        'timestamp': timestamp,
        'language': language,
        'complaint': complaint,
        'category': category,
        'urgency': urgency,
        'department': department,
        'location': location,
        'latitude': latitude,
        'longitude': longitude,
        'status': 'Reported',
        'reporter': reporter
    })

    return jsonify({
        'timestamp': timestamp,
        'category': category.replace('_', ' ').title(),
        'urgency': urgency,
        'department': department,
        'status': 'Reported',
        'latitude': latitude,
        'longitude': longitude,
        'location': location
    })

@app.route('/incidents', methods=['GET'])
def list_incidents():
    limit = request.args.get('limit', default=10, type=int)
    incidents = load_recent_incidents(limit)
    return jsonify({'count': len(incidents), 'incidents': incidents})

@app.route('/incident_summary', methods=['GET'])
def incident_summary():
    incidents = load_recent_incidents(500)
    summary = {
        'total': len(incidents),
        'critical': 0,
        'reported': 0,
        'by_department': {}
    }
    for incident in incidents:
        if incident.get('urgency') == 'Critical urgency':
            summary['critical'] += 1
        if incident.get('status') == 'Reported':
            summary['reported'] += 1
        department = incident.get('department', 'Traffic Control Room')
        summary['by_department'][department] = summary['by_department'].get(department, 0) + 1
    return jsonify(summary)

@app.route('/traffic_map_data', methods=['GET'])
def traffic_map_data():
    incidents = load_recent_incidents(500)
    live_factors = get_live_traffic_factors()
    scored_incidents = []
    level_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

    for incident in incidents:
        traffic_state = traffic_score_for_incident(incident, live_factors)
        level_counts[traffic_state['level']] = level_counts.get(traffic_state['level'], 0) + 1
        scored_incidents.append({
            **incident,
            'traffic_score': traffic_state['score'],
            'traffic_level': traffic_state['level']
        })

    if scored_incidents:
        average_score = round(sum(item['traffic_score'] for item in scored_incidents) / len(scored_incidents), 1)
    else:
        average_score = round((live_factors['congestion_factor'] * 0.6) + (live_factors['vehicle_factor'] * 0.4), 1)

    overall_level = level_from_score(average_score)
    return jsonify({
        'overall_score': average_score,
        'overall_level': overall_level,
        'live_factors': live_factors,
        'level_counts': level_counts,
        'incidents': scored_incidents
    })

@app.route('/traffic_trend', methods=['GET'])
def traffic_trend():
    incidents = load_recent_incidents(500)
    category_counts = {}
    urgency_counts = {'Critical urgency': 0, 'High urgency': 0, 'Medium urgency': 0}

    for incident in incidents:
        category = incident.get('category', 'unknown')
        urgency = incident.get('urgency', 'Medium urgency')
        category_counts[category] = category_counts.get(category, 0) + 1
        urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1

    ranked = sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
    most_reported = ranked[0] if ranked else None
    least_reported = ranked[-1] if ranked else None

    return jsonify({
        'total': len(incidents),
        'category_counts': category_counts,
        'urgency_counts': urgency_counts,
        'most_reported': {
            'category': most_reported[0].replace('_', ' ').title(),
            'count': most_reported[1]
        } if most_reported else None,
        'least_reported': {
            'category': least_reported[0].replace('_', ' ').title(),
            'count': least_reported[1]
        } if least_reported else None
    })

@app.route('/predict_congestion', methods=['POST'])
def predict_congestion():
    data    = request.json
    hour    = int(data.get('hour', 8))
    day     = int(data.get('day', 0))
    month   = int(data.get('month', 6))
    temp    = float(data.get('temp', 285))
    rain    = float(data.get('rain', 0))
    clouds  = float(data.get('clouds', 40))
    weather = data.get('weather', 'Clear')
    try:
        weather_encoded = le_weather.transform([weather])[0]
    except:
        weather_encoded = 0
    features = pd.DataFrame(
        [[hour, day, month, temp, rain, clouds, weather_encoded]],
        columns=['hour','day_of_week','month','temp',
                 'rain_1h','clouds_all','weather_encoded']
    )
    prediction = phase2_model.predict(features)[0]
    advice_map = {
        'low':    'Roads are clear. Good time to travel! 🟢',
        'medium': 'Moderate traffic. Allow extra time. 🟡',
        'high':   'Heavy congestion. Take alternate routes! 🔴'
    }
    timestamp = datetime.utcnow().isoformat() + 'Z'
    append_csv_row(CONGESTION_CSV, [
        'timestamp', 'hour', 'day_of_week', 'month', 'temp', 'rain_1h', 'clouds_all',
        'weather', 'weather_encoded', 'congestion', 'advice'
    ], {
        'timestamp': timestamp,
        'hour': hour,
        'day_of_week': day,
        'month': month,
        'temp': temp,
        'rain_1h': rain,
        'clouds_all': clouds,
        'weather': weather,
        'weather_encoded': weather_encoded,
        'congestion': prediction.upper(),
        'advice': advice_map.get(prediction, '')
    })

    return jsonify({
        'congestion': prediction.upper(),
        'advice':     advice_map.get(prediction, '')
    })

@app.route('/detect_vehicle', methods=['POST'])
def detect_vehicle():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'})
    img    = Image.open(file.stream).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model_cv(tensor)
        _, predicted = output.max(1)
    classes = ['Car detected 🚗', 'Truck detected 🚛']
    advice  = [
        'Light vehicle — normal traffic flow',
        'Heavy vehicle detected — possible congestion ahead'
    ]
    result = classes[predicted.item()]
    timestamp = datetime.utcnow().isoformat() + 'Z'
    append_csv_row(VEHICLE_CSV, [
        'timestamp', 'result', 'advice'
    ], {
        'timestamp': timestamp,
        'result': result,
        'advice': advice[predicted.item()]
    })

    return jsonify({
        'result': result,
        'advice': advice[predicted.item()]
    })

if __name__ == '__main__':
    app.run(debug=True)