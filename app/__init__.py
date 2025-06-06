import os
from flask import Flask
from .routes.predict import bp as predict_bp

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))

# Load model và scaler một lần khi khởi động
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'data', 'scaler.pkl')

try:
    from .utils.model import model, scaler
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    raise

# Register blueprints
app.register_blueprint(predict_bp)

def create_app():
    return app