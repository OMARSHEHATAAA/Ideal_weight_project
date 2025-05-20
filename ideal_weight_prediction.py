
# ⚖️ توقع الوزن المثالي باستخدام الطول والعمر والجنس

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# 📥 تحميل البيانات
df = pd.read_csv("ideal_weight_dataset.csv")

# 👁️‍🗨️ نظرة عامة
print("\n📄 نظرة على البيانات:")
print(df.head())
print("\nℹ️ معلومات:")
print(df.info())

# 🔄 تجهيز البيانات
le = LabelEncoder()
df["الجنس"] = le.fit_transform(df["الجنس"])

# ✂️ تقسيم البيانات
X = df[["الطول_سم", "العمر", "الجنس"]]
y = df["الوزن_كلغ"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚙️ تدريب النموذج
model = LinearRegression()
model.fit(X_train, y_train)

# 🔍 التقييم
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n📊 MSE: {mse:.2f}")
print(f"📈 R^2 Score: {r2:.2f}")

# ✅ تجربة توقع
new_person = pd.DataFrame([[172, 26, 0]], columns=["الطول_سم", "العمر", "الجنس"])
predicted_weight = model.predict(new_person)[0]
print(f"\n✅ الوزن المتوقع لشخص طوله 172، عمره 26، أنثى: {predicted_weight:.2f} كغ")

# 📈 رسم العلاقات
sns.pairplot(df, hue="الجنس")
plt.suptitle("العلاقات بين الخصائص المختلفة", y=1.02)
plt.show()
