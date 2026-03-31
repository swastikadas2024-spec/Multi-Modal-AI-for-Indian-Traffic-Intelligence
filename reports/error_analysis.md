# Error Analysis Report

Generated: 2026-03-31 16:44:51Z

## Top failure cases with explanations

No misclassified examples found in this run.


## Error patterns by location, time, language, weather

### location

| Bucket | Errors | Total | Error Rate |
|---|---:|---:|---:|
| downtown | 0 | 4 | 0.00 |
| north_zone | 0 | 3 | 0.00 |
| south_zone | 0 | 3 | 0.00 |
| suburbs | 0 | 2 | 0.00 |
### time_bucket

| Bucket | Errors | Total | Error Rate |
|---|---:|---:|---:|
| afternoon | 0 | 4 | 0.00 |
| evening | 0 | 4 | 0.00 |
| morning | 0 | 4 | 0.00 |
### language_type

| Bucket | Errors | Total | Error Rate |
|---|---:|---:|---:|
| english_only | 0 | 8 | 0.00 |
| hinglish_mixed | 0 | 4 | 0.00 |
### weather

| Bucket | Errors | Total | Error Rate |
|---|---:|---:|---:|
| clear | 0 | 6 | 0.00 |
| fog | 0 | 2 | 0.00 |
| rain | 0 | 4 | 0.00 |

## Error rate per class

| Class | Total | Errors | Error Rate |
|---|---:|---:|---:|
| accident | 4 | 0 | 0.00 |
| congestion | 5 | 0 | 0.00 |
| potholes | 3 | 0 | 0.00 |

## False positives vs false negatives

| Class | False Positives | False Negatives |
|---|---:|---:|
| accident | 0 | 0 |
| congestion | 0 | 0 |
| potholes | 0 | 0 |

## Root cause analysis

- No misclassifications found. Root cause analysis is not applicable.
- Error distribution plot: reports/figures/error_distribution.png

## Recommendations for improvement

- Increase Hinglish and weather-specific training samples.
- Add confusion-aware hard-negative mining for commonly swapped classes.
- Calibrate confidence threshold by slice rather than using a global threshold.