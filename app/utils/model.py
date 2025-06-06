import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'data', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict(input_data):
    input_scaled = scaler.transform([input_data])
    return model.predict(input_scaled)[0]

def analyze_contributions(input_data):
    feature_names = ['Giới tính', 'Tuổi', 'Trình độ học vấn', 'Hút thuốc hiện tại', 'Số điếu thuốc mỗi ngày',
                     'Thuốc huyết áp', 'Đột quỵ lịch sử', 'Cao huyết áp', 'Tiểu đường', 'Cholesterol toàn phần',
                     'Huyết áp tâm thu', 'Huyết áp tâm trương', 'BMI', 'Nhịp tim', 'Đường huyết']
    input_scaled = scaler.transform([input_data])
    weights = model.coef_[0]
    contributions = {feature_names[i]: abs(input_scaled[0][i] * weights[i]) for i in range(len(input_data))}
    return sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]