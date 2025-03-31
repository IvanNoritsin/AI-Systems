import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

def backward_elimination(X, y, threshold=0.05):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    while True:
        model = sm.OLS(y, X).fit()
        pvalues = model.pvalues.drop('const', errors='ignore')
        if pvalues.empty:
            break
            
        max_pval = pvalues.max()
        if max_pval > threshold:
            feature_to_remove = pvalues.idxmax()
            print(f"Удаляем {feature_to_remove} (p-value: {max_pval:.4f})")
            X = X.drop(feature_to_remove, axis=1)
        else:
            break
    
    print("\nИтоговая модель:")
    print(model.summary())
    return X

data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 
           'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']

data = pd.read_csv(data_path, header=None, names=columns)

print(data.head())
print("\n----------------------------------------------------------------------")

# ---

data = data[data['Height'] > 0]
data['Age'] = data['Rings'] + 1.5
data.drop('Rings', axis=1, inplace=True)

print(data.head())
print("\n----------------------------------------------------------------------")

# ---

X = data.drop('Age', axis=1)
y = data['Age']

print("\nМатрица признаков")
print(X[:10])
print("\nЗависимая переменная")
print(y[:10])
print("\n----------------------------------------------------------------------")

# ---

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), ['Sex'])], 
    remainder='passthrough'
)
X_encoded = ct.fit_transform(X)

print("\nМатрица признаков после обработки категориальных признаков:")
print(X_encoded[:5])
print("\n----------------------------------------------------------------------")

encoder = ct.named_transformers_['encoder']
encoded_features = encoder.get_feature_names_out(['Sex'])
numeric_features = [col for col in X.columns if col != 'Sex']
all_features = np.concatenate([encoded_features, numeric_features])
X_encoded_df = pd.DataFrame(X_encoded, columns=all_features)
X_encoded_df = sm.add_constant(X_encoded_df)

print(X_encoded_df.head())
print("\n----------------------------------------------------------------------\n")

# ---

X_optimal = backward_elimination(X_encoded_df.copy(), y)
y = y.reset_index(drop=True)

# ---

X_train, X_test, y_train, y_test = train_test_split(X_optimal, 
                                                    y, 
                                                    test_size=0.25, 
                                                    random_state=0)

# ---

final_model = sm.OLS(y_train, X_train).fit()
print("\nМодель на обучающих данных:")
print(final_model.summary())

y_pred = final_model.predict(X_test)

print("\nПервые 10 зависимых переменных (тестовая выборка)")
print(y_test[:10])
print("\nПервые 10 зависимых переменных (предсказание)")
print(y_pred[:10])

# ---

plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, line_kws={'color': 'red'})
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение фактических и предсказанных значений')


