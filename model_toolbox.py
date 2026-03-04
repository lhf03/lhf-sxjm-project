# modeling_toolbox.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 设置全局绘图风格（中文字体处理）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def data_preprocessing(file_path):
    """数据预处理：读取、填补、去异常值、标准化"""
    df = pd.read_excel(file_path)
    df = df.fillna(df.mean())
    for col in df.select_dtypes(include=[np.number]).columns:
        mu, sigma = df[col].mean(), df[col].std()
        df = df[(df[col] > mu - 3*sigma) & (df[col] < mu + 3*sigma)]
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def topsis_eval(data):
    """综合评价：熵权法 + TOPSIS"""
    p = data / (data.sum(axis=0) + 1e-9)
    e = - (p * np.log(p + 1e-9)).sum(axis=0) / np.log(len(data))
    weights = (1 - e) / (1 - e).sum()
    weighted_data = data * weights
    best_v, worst_v = weighted_data.max(axis=0), weighted_data.min(axis=0)
    d_plus = np.sqrt(((weighted_data - best_v)**2).sum(axis=1))
    d_minus = np.sqrt(((weighted_data - worst_v)**2).sum(axis=1))
    return d_minus / (d_plus + d_minus)

def plot_academic_style(data, x_col, y_col, title):
    """学术绘图：高分辨率回归图"""
    plt.figure(figsize=(8, 5), dpi=300)
    sns.regplot(data=data, x=x_col, y=y_col, scatter_kws={'alpha':0.5})
    plt.title(title)
    plt.show()

def quick_rf_model(X, y):
    """随机森林：回归预测与特征重要性"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    print(f"模型 R2 得分: {model.score(X_test, y_test):.4f}")
    return model, model.feature_importances_

def sensitivity_analysis(model, base_input, param_index):
    """灵敏度分析：观察单一变量波动对结果的影响"""
    scales = np.linspace(0.8, 1.2, 11)
    results = []
    for s in scales:
        test_in = base_input.copy().astype(float)
        test_in[param_index] *= s
        results.append(model.predict(test_in.reshape(1, -1))[0])
    plt.plot(scales, results, 'o-')
    plt.title("灵敏度分析")
    plt.show()
