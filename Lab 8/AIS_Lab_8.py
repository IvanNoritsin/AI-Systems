import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"

data = pd.read_csv(data_path)

print(data.head())
print("\n----------------------------------------------------------------------")

data['area'] = np.log1p(data['area'])
data = data[(data['area'] > 0) & (data['area'] < 50)].copy()
data.reset_index(drop=True, inplace=True)

print(data.head())
print("\n----------------------------------------------------------------------")

# ---

X = data.drop('area', axis=1)
y = data['area']

print("\nМатрица признаков")
print(X[:10])
print("\nЗависимая переменная")
print(y[:10])
print("\n----------------------------------------------------------------------")

# ---

ct = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(drop='first'), [2, 3])
    ],
    remainder="passthrough"
)

X_encoded = ct.fit_transform(X)

print("\nМатрица признаков после обработки категориальных признаков:")
print(X_encoded[:5])
print("\n----------------------------------------------------------------------")

# ---

X_train, X_test, y_train, y_test = train_test_split(X_encoded, 
                                                    y, 
                                                    test_size=0.25, 
                                                    random_state=42)

# ---

degrees = [1, 2, 3, 4]

plt.figure(figsize=(12, 10))

for i, degree in enumerate(degrees, 1):
    poly_reg = PolynomialFeatures(degree=degree)
    X_train_poly = poly_reg.fit_transform(X_train)
    X_test_poly = poly_reg.transform(X_test)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_poly, y_train)
    y_test_pred = lin_reg.predict(X_test_poly)

    plt.subplot(2, 2, i)
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.9, color='blue', label="Предсказания")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Идеальное предсказание")
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'Полиномиальная регрессия (degree={degree})')
    plt.legend()

plt.tight_layout()
plt.show()