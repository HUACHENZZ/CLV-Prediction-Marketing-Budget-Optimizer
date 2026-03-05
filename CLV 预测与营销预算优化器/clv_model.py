# ==========================================
# Customer Lifetime Value Prediction Model
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 连接 MySQL 数据库
# ==========================================
# 请修改为你的 MySQL 密码
engine = create_engine('mysql+pymysql://root:030622@localhost:3306/olist_analytics')

print("📊 Loading data from MySQL...")
query = "SELECT * FROM v_customer_features"
df = pd.read_sql(query, engine)
print(f"✅ Loaded {len(df)} customers")

# ==========================================
# 2. 数据预处理
# ==========================================
print("\n🔧 Preprocessing data...")

# 处理空值
df['avg_purchase_interval_days'] = df['avg_purchase_interval_days'].fillna(df['avg_purchase_interval_days'].median())
df = df.dropna()

# 创建目标变量：预测未来价值 (简化模拟)
# 实际项目中应该用历史数据训练，这里用 monetary * 系数模拟
np.random.seed(42)
df['target_clv'] = df['monetary'] * np.random.uniform(1.5, 3.0, len(df))

# 选择特征
features = ['frequency', 'recency_days', 'monetary', 'avg_order_value',
            'customer_lifespan', 'avg_purchase_interval_days']
X = df[features]
y = df['target_clv']

# ==========================================
# 3. 训练预测模型
# ==========================================
print("\n🤖 Training Random Forest Model...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📈 Model Performance:")
print(f"   RMSE: ${rmse:.2f}")
print(f"   R² Score: {r2:.2f}")
print(f"   MAE: ${mae:.2f}")

# ==========================================
# 4. SHAP 可解释性分析 (统计学亮点)
# ==========================================
print("\n🔍 Running SHAP analysis...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 保存 SHAP 值供 Power BI 使用
shap_df = pd.DataFrame(shap_values, columns=[f'shap_{col}' for col in features])
shap_df['customer_unique_id'] = X_test.index.tolist()
shap_df.to_csv('shap_values.csv', index=False)

# 保存特征重要性
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
feature_importance.to_csv('feature_importance.csv', index=False)

print("✅ SHAP analysis saved to shap_values.csv")

# ==========================================
# 5. 生成预测结果
# ==========================================
print("\n📋 Generating predictions...")

df['predicted_clv'] = model.predict(X)
df['clv_segment'] = pd.qcut(df['predicted_clv'], q=4, labels=['Low', 'Medium', 'High', 'Champion'])

# 保存完整结果
df.to_csv('customer_clv_predictions.csv', index=False)
print("✅ Predictions saved to customer_clv_predictions.csv")

# ==========================================
# 6. 创建可视化图表
# ==========================================
print("\n📊 Generating visualizations...")

# 图 1: 特征重要性
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance for CLV Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)

# 图 2: CLV 分布
plt.figure(figsize=(10, 6))
plt.hist(df['predicted_clv'], bins=50, edgecolor='black')
plt.xlabel('Predicted CLV ($)')
plt.title('Customer Lifetime Value Distribution')
plt.tight_layout()
plt.savefig('clv_distribution.png', dpi=300)

# 图 3: 各 segment 客户数量
plt.figure(figsize=(10, 6))
segment_counts = df['clv_segment'].value_counts()
plt.bar(segment_counts.index, segment_counts.values)
plt.xlabel('CLV Segment')
plt.ylabel('Number of Customers')
plt.title('Customer Distribution by CLV Segment')
plt.tight_layout()
plt.savefig('segment_distribution.png', dpi=300)

print("✅ Visualizations saved")

# ==========================================
# 7. 输出业务洞察摘要
# ==========================================
print("\n" + "="*50)
print("📊 BUSINESS INSIGHTS SUMMARY")
print("="*50)
print(f"Total Customers Analyzed: {len(df):,}")
print(f"Average Predicted CLV: ${df['predicted_clv'].mean():.2f}")
print(f"Top Feature Driver: {feature_importance.iloc[0]['feature']}")
print(f"Champion Segment (Top 25%): {len(df[df['clv_segment']=='Champion']):,} customers")
print(f"Champion Segment Revenue Contribution: ${df[df['clv_segment']=='Champion']['predicted_clv'].sum():.2f}")
print("="*50)

print("\n✅ All tasks completed successfully!")
