import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Đọc dữ liệu đã xử lý
df = pd.read_csv('../data/framingham_processed.csv')

# Chọn đặc trưng đầu vào (X) và biến mục tiêu (y)
X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']

# Số lần chạy để đánh giá
n_runs = 5
accuracies = []
auc_scores = []

# Vòng lặp để chạy mô hình nhiều lần với các tập huấn luyện/kiểm tra ngẫu nhiên
for i in range(n_runs):
    print(f"\nLần chạy thứ {i+1}:")

    # Chia dữ liệu thành tập huấn luyện và kiểm tra (80/20) mà không cố định random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        stratify=y)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện mô hình Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Dự đoán
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    accuracies.append(accuracy)
    auc_scores.append(auc)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("AUC Score:", auc)

# Tính trung bình và độ lệch chuẩn của các chỉ số
print("\nKết quả trung bình sau", n_runs, "lần chạy:")
print("Accuracy trung bình:", np.mean(accuracies))
print("Độ lệch chuẩn Accuracy:", np.std(accuracies))
print("AUC trung bình:", np.mean(auc_scores))
print("Độ lệch chuẩn AUC:", np.std(auc_scores))

# Vẽ đường cong ROC cho lần chạy cuối cùng
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve - Logistic Regression (Lần chạy cuối)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()


# Lưu mô hình và scaler của lần chạy cuối
joblib.dump(model, '../data/model.pkl')
joblib.dump(scaler, '../data/scaler.pkl')
print("Mô hình và scaler của lần chạy cuối đã được lưu vào 'model.pkl' và 'scaler.pkl'")