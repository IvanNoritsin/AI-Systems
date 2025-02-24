import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"

columns = [
    "Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"
]

data = pd.read_csv(data_path, names=columns)
data.drop(columns=["Id"], inplace=True)

# ----

X_train = data[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
y_train = data['Type']

K = 6

knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train.values, y_train)

#X_test = np.array([[1.51634, 14.9, 0.0, 2.3, 73.47, 0.0, 8.33, 1.4, 0.0]])
X_test = np.array([[1.52137, 13.84, 3.74, 2.7, 72.64, 0.67, 7.53, 0.0, 0.0]])
target = knn.predict(X_test)
print("Данный набор признаков был отнесён к классу:", target)

# ----

X_train, X_holdout, y_train, y_holdout = train_test_split(data[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']], 
                                                          data['Type'],
                                                          test_size=0.2,
                                                          random_state=17,
                                                          stratify=data['Type'])

k_list = list(range(1,50))
accuracy_array = []

for K in k_list:
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_holdout)
    accuracy = accuracy_score(y_holdout, knn_pred)
    accuracy_array.append(accuracy)
    
plt.figure(1)
plt.plot(k_list, accuracy_array)
plt.xlabel('Количество соседей (K)');
plt.ylabel('Точность классификатора')
    
# ----

cv_scores = []

for K in k_list:
    knn = KNeighborsClassifier(n_neighbors=K)
    scores = cross_val_score(knn, 
                             data[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']],
                             data['Type'],
                             cv=9,
                             scoring='accuracy')
    cv_scores.append(scores.mean())
    
MSE = [1-x for x in cv_scores]

plt.figure(2)
plt.plot(k_list, MSE)
plt.xlabel('Количество соседей (K)');
plt.ylabel('Ошибка классификации (MSE)')

k_min = min(MSE)

all_k_min = []
for i in range(len(MSE)):
    if MSE[i] <= k_min:
        all_k_min.append(k_list[i])
        
print('Оптимальные значения K: ', all_k_min)

# ----

X = data[['Ca', 'Na']]
y = data['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17, stratify=y)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

x_min, x_max = X.iloc[:, 0].min() - 0.1, X.iloc[:, 0].max() + 0.1
y_min, y_max = X.iloc[:, 1].min() - 0.1, X.iloc[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(3)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Pastel1')
sns.scatterplot(x=X_train.iloc[:, 0], y=X_train.iloc[:, 1], hue=y_train, palette='Dark2', edgecolor='k')
plt.xlabel('Ca')
plt.ylabel('Na')