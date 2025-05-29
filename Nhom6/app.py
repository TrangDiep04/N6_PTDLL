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
        male = int(request.form['male'])
        age = float(request.form['age'])
        education = float(request.form['education'])
        currentSmoker = int(request.form['currentSmoker'])
        cigsPerDay = float(request.form['cigsPerDay'])
        BPMeds = int(request.form['BPMeds'])
        prevalentStroke = int(request.form['prevalentStroke'])
        prevalentHyp = int(request.form['prevalentHyp'])
        diabetes = int(request.form['diabetes'])
        totChol = float(request.form['totChol'])
        sysBP = float(request.form['sysBP'])
        diaBP = float(request.form['diaBP'])
        BMI = float(request.form['BMI'])
        heartRate = float(request.form['heartRate'])
        glucose = float(request.form['glucose'])

        # Tạo mảng đầu vào với tất cả các đặc trưng (thứ tự khớp với dữ liệu huấn luyện)
        input_data = np.array([
            [male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke,
             prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]
        ])
        input_scaled = scaler.transform(input_data)

        # Dự đoán nhị phân (0 hoặc 1)
        binary_prediction = model.predict(input_scaled)[0]
        # Dự đoán xác suất (%)
        probability = model.predict_proba(input_scaled)[0][1] * 100
        probability = round(probability, 2)

        return render_template('index.html', binary_prediction=binary_prediction, probability=probability)
    except Exception as e:
        return render_template('index.html', error=f"Đã có lỗi xảy ra: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)