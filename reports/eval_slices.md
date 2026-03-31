# Slice-Based Evaluation

Generated: 2026-03-31 16:44:40Z

## Slice-based evaluation table

| Slice | Count | F1 | Precision | Recall | Notes |
|---|---:|---:|---:|---:|---|
| morning | 4 | 1.00 | 1.00 | 1.00 | Good performance |
| afternoon | 4 | 1.00 | 1.00 | 1.00 | Good performance |
| evening | 4 | 1.00 | 1.00 | 1.00 | Good performance |
| north_zone | 3 | 1.00 | 1.00 | 1.00 | Good performance |
| south_zone | 3 | 1.00 | 1.00 | 1.00 | Good performance |
| downtown | 4 | 1.00 | 1.00 | 1.00 | Good performance |
| suburbs | 2 | 1.00 | 1.00 | 1.00 | Good performance |
| clear_weather | 6 | 1.00 | 1.00 | 1.00 | Good performance |
| rain_weather | 4 | 1.00 | 1.00 | 1.00 | Good performance |
| fog_weather | 2 | 1.00 | 1.00 | 1.00 | Good performance |
| english_only | 8 | 1.00 | 1.00 | 1.00 | Good performance |
| hinglish_mixed | 4 | 1.00 | 1.00 | 1.00 | Good performance |

## Per-class metrics by slice

### morning

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| accident | 0.00 | 0.00 | 0.00 | 0 |
| congestion | 1.00 | 1.00 | 1.00 | 1 |
| potholes | 1.00 | 1.00 | 1.00 | 3 |

Confusion matrix: reports/figures\confusion_matrix_morning.png

### afternoon

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| accident | 1.00 | 1.00 | 1.00 | 2 |
| congestion | 1.00 | 1.00 | 1.00 | 2 |
| potholes | 0.00 | 0.00 | 0.00 | 0 |

Confusion matrix: reports/figures\confusion_matrix_afternoon.png

### evening

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| accident | 1.00 | 1.00 | 1.00 | 2 |
| congestion | 1.00 | 1.00 | 1.00 | 2 |
| potholes | 0.00 | 0.00 | 0.00 | 0 |

Confusion matrix: reports/figures\confusion_matrix_evening.png

### north_zone

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| accident | 1.00 | 1.00 | 1.00 | 1 |
| congestion | 1.00 | 1.00 | 1.00 | 1 |
| potholes | 1.00 | 1.00 | 1.00 | 1 |

Confusion matrix: reports/figures\confusion_matrix_north_zone.png

### south_zone

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| accident | 1.00 | 1.00 | 1.00 | 2 |
| congestion | 0.00 | 0.00 | 0.00 | 0 |
| potholes | 1.00 | 1.00 | 1.00 | 1 |

Confusion matrix: reports/figures\confusion_matrix_south_zone.png

### downtown

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| accident | 1.00 | 1.00 | 1.00 | 1 |
| congestion | 1.00 | 1.00 | 1.00 | 3 |
| potholes | 0.00 | 0.00 | 0.00 | 0 |

Confusion matrix: reports/figures\confusion_matrix_downtown.png

### suburbs

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| accident | 0.00 | 0.00 | 0.00 | 0 |
| congestion | 1.00 | 1.00 | 1.00 | 1 |
| potholes | 1.00 | 1.00 | 1.00 | 1 |

Confusion matrix: reports/figures\confusion_matrix_suburbs.png

### clear_weather

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| accident | 0.00 | 0.00 | 0.00 | 0 |
| congestion | 1.00 | 1.00 | 1.00 | 3 |
| potholes | 1.00 | 1.00 | 1.00 | 3 |

Confusion matrix: reports/figures\confusion_matrix_clear_weather.png

### rain_weather

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| accident | 1.00 | 1.00 | 1.00 | 2 |
| congestion | 1.00 | 1.00 | 1.00 | 2 |
| potholes | 0.00 | 0.00 | 0.00 | 0 |

Confusion matrix: reports/figures\confusion_matrix_rain_weather.png

### fog_weather

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| accident | 1.00 | 1.00 | 1.00 | 2 |
| congestion | 0.00 | 0.00 | 0.00 | 0 |
| potholes | 0.00 | 0.00 | 0.00 | 0 |

Confusion matrix: reports/figures\confusion_matrix_fog_weather.png

### english_only

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| accident | 1.00 | 1.00 | 1.00 | 3 |
| congestion | 1.00 | 1.00 | 1.00 | 4 |
| potholes | 1.00 | 1.00 | 1.00 | 1 |

Confusion matrix: reports/figures\confusion_matrix_english_only.png

### hinglish_mixed

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| accident | 1.00 | 1.00 | 1.00 | 1 |
| congestion | 1.00 | 1.00 | 1.00 | 1 |
| potholes | 1.00 | 1.00 | 1.00 | 2 |

Confusion matrix: reports/figures\confusion_matrix_hinglish_mixed.png

## Recommendations

- Retrain on Hinglish-heavy and weather-tagged complaints.
- Add augmentation for rain/fog phrasing and noisy social text.
- Monitor evening slice drift and recalibrate confidence thresholds.