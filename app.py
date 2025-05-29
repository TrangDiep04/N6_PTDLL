from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model và scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        input_data = [
            int(request.form['male']),
            float(request.form['age']),
            float(request.form['education']),
            int(request.form['currentSmoker']),
            float(request.form['cigsPerDay']),
            int(request.form['BPMeds']),
            int(request.form['prevalentStroke']),
            int(request.form['prevalentHyp']),
            int(request.form['diabetes']),
            float(request.form['totChol']),
            float(request.form['sysBP']),
            float(request.form['diaBP']),
            float(request.form['BMI']),
            float(request.form['heartRate']),
            float(request.form['glucose'])
        ]

        input_scaled = scaler.transform([input_data])

        # Dự đoán nhị phân và xác suất
        binary_prediction = model.predict(input_scaled)[0]
        probability = round(model.predict_proba(input_scaled)[0][1] * 100, 2)

        # Phân tích yếu tố đóng góp
        contributions = analyze_contributions(input_data)

        recommendations = generate_recommendations(input_data)

        return render_template('index.html',
                               binary_prediction=binary_prediction,
                               probability=probability,
                               contributions=contributions,
                               recommendations=recommendations)
    except Exception as e:
        return render_template('index.html', error=f"Đã có lỗi xảy ra: {str(e)}")


def analyze_contributions(input_data):
    # Giả sử trọng số từ mô hình
    weights = model.coef_[0]
    contributions = {f'Feature {i}': input_data[i] * weights[i] for i in range(len(input_data))}
    return sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]


def generate_recommendations(input_data):
    recommendations = []
    if input_data[10] > 140:  # sysBP
        recommendations.append("Giảm muối trong khẩu phần ăn.")
    if input_data[9] > 240:  # totChol
        recommendations.append("Giảm ăn chất béo bão hòa.")
    if input_data[4] > 0:  # cigsPerDay
        recommendations.append("Cố gắng giảm số lượng thuốc lá.")
    if input_data[12] > 30:  # BMI
        recommendations.append("Thực hiện chế độ ăn uống lành mạnh và tập thể dục.")

    return recommendations


if __name__ == '__main__':
    app.run(debug=True)