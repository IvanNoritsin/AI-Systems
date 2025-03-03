import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import tree


data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"

columns = [
    "Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"
]

data = pd.read_csv(data_path, names=columns)
data.drop(columns=["Id"], inplace=True)

# ---

X_train, X_holdout, y_train, y_holdout = train_test_split(data[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']],
                                                          data['Type'],
                                                          test_size=0.3, 
                                                          random_state=12,
                                                          stratify=data['Type'])

tree_model = DecisionTreeClassifier(max_depth=5,
                              random_state=21,
                              max_features=2)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_holdout)
accuracy = accuracy_score(y_holdout, tree_pred)
print("Оценка точности классификатора методом hold-out:", accuracy)

tree.export_graphviz(tree_model,
                     feature_names=data[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']].columns,
                     class_names=[str(c) for c in data['Type'].unique()],
                     out_file='glass_tree_1.dot',
                     filled=True, rounded=True)

# ---

d_list = list(range(1,20))
cv_scores = []

for d in d_list:
    tree_model = DecisionTreeClassifier(max_depth=d,
                                  random_state=21,
                                  max_features=2)
    scores = cross_val_score(tree_model, 
                             data[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']],
                             data['Type'],
                             cv=9,
                             scoring='accuracy')
    cv_scores.append(scores.mean())
    
MSE = [1-x for x in cv_scores]

plt.figure(1)
plt.plot(d_list, MSE)
plt.xlabel('Макс. глубина дерева (max_depth)');
plt.ylabel('Ошибка классификации (MSE)')

d_min = min(MSE)
all_d_min = []
for i in range(len(MSE)):
    if MSE[i] <= d_min:
        all_d_min.append(d_list[i])
        
print('Оптимальные значения max_depth: ', all_d_min)

# --- 

f_list = list(range(1, 10))
cv_scores_features = []

for f in f_list:
    tree_model = DecisionTreeClassifier(max_depth=5,
                                  random_state=21,
                                  max_features=f)
    scores = cross_val_score(tree_model, 
                             data[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']],
                             data['Type'],
                             cv=9,
                             scoring='accuracy')
    cv_scores_features.append(scores.mean())
    
MSE_features = [1-x for x in cv_scores_features]

plt.figure(2)
plt.plot(f_list, MSE_features)
plt.xlabel('Количество признаков разделения дерева (max_features)');
plt.ylabel('Ошибка классификации (MSE)')

f_min = min(MSE_features)
all_f_min = []
for i in range(len(MSE_features)):
    if MSE_features[i] <= f_min:
        all_f_min.append(f_list[i])
        
print('Оптимальные значения max_features: ', all_f_min)

# ---

dtc = DecisionTreeClassifier(max_depth=10, random_state=21, max_features=2)
tree_params = { 'max_depth': range(1,20), 'max_features': range(1,10) }
tree_grid = GridSearchCV(dtc, tree_params, cv=9, verbose=True, n_jobs=-1)
tree_grid.fit(data[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']], data['Type'])
print('Лучшее сочетание параметров: ', tree_grid.best_params_)
print('Лучшие баллы cross validation: ', tree_grid.best_score_)

tree.export_graphviz(tree_grid.best_estimator_,
                                feature_names=data[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']].columns,
                                class_names=[str(c) for c in data['Type'].unique()],
                                out_file='glass_tree_2.dot',
                                filled=True, rounded=True)

# ---

X = data[['Ca', 'Na']]
y = data['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17, stratify=y)
tree_model = DecisionTreeClassifier(max_depth=12, random_state=21, max_features=3)
tree_model.fit(X_train, y_train)

x_min, x_max = X.iloc[:, 0].min() - 0.1, X.iloc[:, 0].max() + 0.1
y_min, y_max = X.iloc[:, 1].min() - 0.1, X.iloc[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = tree_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(3)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Pastel1')
sns.scatterplot(x=X_train.iloc[:, 0], y=X_train.iloc[:, 1], hue=y_train, palette='Dark2', edgecolor='k')
plt.xlabel('Ca')
plt.ylabel('Na')