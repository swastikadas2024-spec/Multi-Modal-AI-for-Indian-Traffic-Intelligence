SmartTraffic India 

An AI-powered smart traffic analysis system combining NLP, Machine Learning, and Computer Vision to address urban traffic challenges in India.
Phase 1 — NLP Complaint Classifier
- Classifies traffic complaints into 5 categories
- Model: Gradient Boosting with TF-IDF Vectorizer
- Dataset: 1,482 real traffic complaints
- Accuracy: 91.25% | F1 Score: 91.30%
Phase 2 — Congestion Predictor
- Predicts traffic congestion from time and weather
- Model: Random Forest Classifier
- Dataset: 48,204 real hourly traffic records (UCI)
- Accuracy: 93.50% | F1 Score: 93.46%
Phase 3 — Vehicle Detector
- Detects vehicle type from road images
- Model: ResNet-18 (Transfer Learning, PyTorch)
- Dataset: 10,000 images (CIFAR-10)
- Accuracy: 91.55%
Tech Stack
- Python, Flask, Scikit-learn, PyTorch
- TF-IDF, Gradient Boosting, Random Forest, ResNet-18
- HTML, CSS, JavaScript
Dataset Sources
- Traffic complaints: Hugging Face (multiclass sentiment dataset)
- Congestion data: UCI Metro Interstate Traffic Volume Dataset
- Vehicle images: CIFAR-10 Dataset
