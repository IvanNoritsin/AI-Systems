import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"

columns = [
    "Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"
]

data = pd.read_csv(data_path, names=columns)
data.drop(columns=["Id"], inplace=True)

print(data.head())
print("\n----------------------------------------------------------------------")

# ---

X = data.iloc[:, :-1].values
y = data.iloc[:, 9].values

print("\nМатрица признаков")
print(X)
print("\nЗависимая переменная")
print(y)
print("\n----------------------------------------------------------------------")

# ---

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 0:9])
X_without_nan = X.copy()
X_without_nan[:, 0:9] = imputer.transform(X[:, 0:9])

print("\nМатрица признаков после заполнения пропусков")
print(X_without_nan)
print("\n----------------------------------------------------------------------")

# ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_without_nan[:, 0:9])

print("\nМатрица признаков после масштабирования")
print(X_scaled)
print("\n----------------------------------------------------------------------")

# ---

X_data = pd.DataFrame(X_scaled, columns=columns[1:-1])
y_data = pd.DataFrame(y, columns=["Type"])
data_new = pd.concat([X_data, y_data], axis=1)
print(" ")
print(data_new.head())
print("\n----------------------------------------------------------------------")

# ---

X_train, X_test, y_train, y_test = train_test_split(X_scaled, 
                                                    y, 
                                                    test_size=0.3, 

                                                    random_state=24)


X_train_df = pd.DataFrame(X_train, columns=columns[1:-1])
y_train_df = pd.DataFrame(y_train, columns=["Type"])

X_test_df = pd.DataFrame(X_test, columns=columns[1:-1])
y_test_df = pd.DataFrame(y_test, columns=["Type"])

print("\nОбучающая выборка (X_train):")
print(X_train_df.head())
print("\nЦелевая переменная обучающей выборки (y_train):")
print(y_train_df.head())

print("\nТестовая выборка (X_test):")
print(X_test_df.head())
print("\nЦелевая переменная тестовой выборки (y_test):")
print(y_test_df.head())
