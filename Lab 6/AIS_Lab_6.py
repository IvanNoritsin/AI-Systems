import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"

data = pd.read_csv(data_path)

print(data.head())
print("\n----------------------------------------------------------------------")

data = data[['temp', 'area']]

print(data.head())
print("\n----------------------------------------------------------------------")

# ---

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

print("\nМатрица признаков")
print(X[:10])
print("\nЗависимая переменная")
print(y[:10])
print("\n----------------------------------------------------------------------")

# ---
'''
ct = ColumnTransformer(transformers=[
    ("encoder", OneHotEncoder(), [2, 3])
], remainder="passthrough")

X = ct.fit_transform(X)

print("\nМатрица признаков после обработки категориальных признаков:")
print(X[:10])
print("\n----------------------------------------------------------------------")
'''
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

print("\nПервые 10 зависимых переменных (тестовая выборка)")
print(y_test[:10])
print("\nПервые 10 зависимых переменных (предсказание)")
print(y_pred[:10])

# ---

plt.scatter(X_test, y_test, color='blue', label='Данные')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Линия регрессии')
plt.xlabel('Температура (temp)')
plt.ylabel('Логарифм площади пожара (log(area + 1))')
plt.title('Одномерная линейная регрессия: temp vs area')
plt.legend()