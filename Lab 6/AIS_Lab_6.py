import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"

data = pd.read_csv(data_path)

print(data.head())
print("\n----------------------------------------------------------------------")

# ---

X = data.iloc[:, :-1].values
y = data.iloc[:, 12].values

print("\nМатрица признаков")
print(X[:10])
print("\nЗависимая переменная")
print(y[:10])
print("\n----------------------------------------------------------------------")

# ---

ct = ColumnTransformer(transformers=[
    ("encoder", OneHotEncoder(), [2, 3])
], remainder="passthrough")

X = ct.fit_transform(X)

print("\nМатрица признаков после обработки категориальных признаков:")
print(X[:10])
print("\n----------------------------------------------------------------------")

# ---

y = np.log1p(y)

# ---

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.25, 
                                                    random_state = 0) 

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print("\nПервые 10 зависимых перем")
print(y_test[:10])
print(y_pred[:10])

# ---

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Линия y = x
plt.xlabel("Фактические значения (y_test)")
plt.ylabel("Предсказанные значения (y_pred)")
plt.title("Предсказанные vs Фактические значения")

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(y_test)), y_test, label="Фактические значения", marker='o')
plt.plot(np.arange(len(y_pred)), y_pred, label="Предсказанные значения", marker='x')
plt.xlabel("Индекс наблюдения")
plt.ylabel("Значение")
plt.title("Фактические и предсказанные значения")
plt.legend()
