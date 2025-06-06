from flask import Blueprint, request, render_template
from ..utils.model import predict, analyze_contributions
from ..utils.advice import generate_recommendations
from ..utils.model import model, scaler

bp = Blueprint('predict', __name__)

@bp.route('/')
def home():
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict_route():
    try:
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

        binary_prediction = predict(input_data)
        probability = round(model.predict_proba(scaler.transform([input_data]))[0][1] * 100, 2)
        contributions = analyze_contributions(input_data)
        recommendations = generate_recommendations(input_data)

        return render_template('index.html',
                              binary_prediction=binary_prediction,
                              probability=probability,
                              contributions=contributions,
                              recommendations=recommendations,
                              input_data=input_data)
    except Exception as e:
        return render_template('index.html', error=f"Đã có lỗi xảy ra: {str(e)}", input_data=request.form.to_dict())