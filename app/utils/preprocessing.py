import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Định dạng đường viền
border = "=========================================================="

# Đọc dữ liệu
df = pd.read_csv('../../data/framingham.csv')

# In ra số liệu hàng, cột
print("-----------------SỐ LIỆU HÀNG - CỘT------------------")
print(df.shape)
print(border)

# Số lượng giá trị khuyết thiếu của mỗi đặc trưng
print("-----------------SỐ LƯỢNG GIÁ TRỊ KHUYẾT THIẾU-------------------")
print(df.isna().sum())
print(border)

# Tạo các chỉ số thống kê mô tả
df_count = df.count()
df_min = df.min()
df_max = df.max()
df_mean = df.mean()
df_median = df.median()
df_q1 = df.quantile(0.25)
df_q2 = df.quantile(0.5)
df_q3 = df.quantile(0.75)
df_iqr = df_q3 - df_q1
df_var = df.var()
df_std = df.std()

# Hàm thống kê sơ lược dữ liệu
def descriptive(df_count, df_min, df_max, df_median, df_mean, df_q1, df_q2, df_q3, df_iqr, df_var, df_std):
    data = {'Count': [i for i in df_count],
            'Min': [i for i in df_min],
            'Max': [i for i in df_max],
            'Median': [i for i in df_median],
            'Mean': [i for i in df_mean],
            'Q1': [i for i in df_q1],
            'Q2': [i for i in df_q2],
            'Q3': [i for i in df_q3],
            'IQR': [i for i in df_iqr],
            'Variance': [i for i in df_var],
            'Std': [i for i in df_std]
           }
    df1 = pd.DataFrame(data)
    df1.index = df.keys()
    data_complete = df1.transpose()
    print(data_complete.to_string())

print("--------------------------BẢNG THỐNG KÊ MÔ TẢ----------------------------")
descriptive(df_count, df_min, df_max, df_median, df_mean, df_q1, df_q2, df_q3, df_iqr, df_var, df_std)
print(border)

# Hàm vẽ biểu đồ Histogram
def draw_histograms(dataframe, features, rows, cols):
    fig = plt.figure(figsize=(15, 15))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows, cols, i + 1)
        dataframe[feature].hist(bins=20, ax=ax, facecolor='pink', edgecolor='blue')
        ax.set_title(feature, color='Black')
        ax.grid(False)
    fig.tight_layout()
    plt.show()

draw_histograms(df, df.columns[:16], 6, 3)

# Xử lý dữ liệu khuyết thiếu
df['education'].fillna(df['education'].mode()[0], inplace=True)
df['BPMeds'].fillna(df['BPMeds'].mode()[0], inplace=True)
df['cigsPerDay'].fillna(df['cigsPerDay'].mean(), inplace=True)
df['totChol'].fillna(df['totChol'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].mean(), inplace=True)
df['heartRate'].fillna(df['heartRate'].mean(), inplace=True)
df['glucose'].fillna(df['glucose'].mean(), inplace=True)

# Dữ liệu sau khi xử lý khuyết thiếu
print("-----------------SỐ LƯỢNG GIÁ TRỊ KHUYẾT THIẾU SAU XỬ LÝ-------------------")
print(df.isna().sum())
print(border)

# Hàm vẽ Boxplot
def draw_boxplots(dataframe, features, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.subplots_adjust(hspace=1)
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.boxplot(x=dataframe[feature], ax=axes[i])
        axes[i].set_title(feature, color='Black')
        axes[i].set_xlabel("")
        axes[i].grid(False)
    plt.show()

# Vẽ boxplot trước khi xử lý outliers
draw_boxplots(df, df.columns[:16], 6, 3)

# Hàm phát hiện các giá trị outliers bằng Z-Score
def detect_outliers_zscore(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    outliers = z_scores > threshold
    return outliers

# Hàm xử lý các giá trị outliers bằng Z-Score
def handle_outliers_zscore(df, column, threshold=3, replace_value=None):
    outlier = detect_outliers_zscore(df, column, threshold)
    df_processed = df.copy()
    if replace_value is not None:
        df_processed.loc[outlier, column] = replace_value
    return df_processed

# Các cột cần xử lý outliers
cot_can_xu_ly_zscore = ['cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

# Xử lý outliers bằng phương pháp Z-score
for cot in cot_can_xu_ly_zscore:
    df = handle_outliers_zscore(df.copy(), cot, replace_value=np.nan)
    df[cot] = df[cot].interpolate()

# Vẽ boxplot sau khi xử lý outliers
draw_boxplots(df, df.columns[:16], 6, 3)

# Lưu dữ liệu đã xử lý vào file CSV mới
df.to_csv('framingham_processed.csv', index=False)
print("Dữ liệu đã xử lý được lưu vào 'framingham_processed.csv'")
print(border)