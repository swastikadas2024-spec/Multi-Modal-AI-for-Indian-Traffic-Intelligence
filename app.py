from flask import Flask, render_template, request, jsonify
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

app = Flask(__name__)

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

# routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify_complaint', methods=['POST'])
def classify_complaint():
    complaint = request.json.get('complaint', '')
    vec = phase1_vectorizer.transform([complaint])
    category = phase1_model.predict(vec)[0]
    urgency_map = {
        'congestion':       'High urgency — immediate action needed',
        'signal_issue':     'Medium urgency — signal maintenance required',
        'road_damage':      'High urgency — safety hazard',
        'public_transport': 'Medium urgency — service improvement needed',
        'road_safety':      'Critical urgency — lives at risk'
    }
    return jsonify({
        'category': category.replace('_', ' ').title(),
        'urgency':  urgency_map.get(category, 'Medium urgency')
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
    return jsonify({
        'result': classes[predicted.item()],
        'advice': advice[predicted.item()]
    })

if __name__ == '__main__':
    app.run(debug=True)