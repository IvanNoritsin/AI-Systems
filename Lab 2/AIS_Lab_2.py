import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"

columns = [
    "Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"
]

data = pd.read_csv(data_path, names=columns)
print(data.head())

print(" ")

data.info()

plt.figure(figsize=(10, 6))
plt.subplots_adjust(wspace=0.6, hspace=0.6)
plt.subplot(3, 3, 1)
data['RI'].hist(bins=30, edgecolor='black')
plt.title("Коэффициент преломления")
plt.xlabel("вес. %")
plt.ylabel("Количество записей")

plt.subplot(3, 3, 2)
data['Na'].hist(bins=30, edgecolor='black')
plt.title("Натрий")
plt.xlabel("вес. %")
plt.ylabel("Количество записей")

plt.subplot(3, 3, 3)
data['Mg'].hist(bins=30, edgecolor='black')
plt.title("Магний")
plt.xlabel("вес. %")
plt.ylabel("Количество записей")

plt.subplot(3, 3, 4)
data['Al'].hist(bins=30, edgecolor='black')
plt.title("Алюминий")
plt.xlabel("вес. %")
plt.ylabel("Количество записей")

plt.subplot(3, 3, 5)
data['Si'].hist(bins=30, edgecolor='black')
plt.title("Кремний")
plt.xlabel("вес. %")
plt.ylabel("Количество записей")

plt.subplot(3, 3, 6)
data['K'].hist(bins=30, edgecolor='black')
plt.title("Калий")
plt.xlabel("вес. %")
plt.ylabel("Количество записей")

plt.subplot(3, 3, 7)
data['Ca'].hist(bins=30, edgecolor='black')
plt.title("Кальций")
plt.xlabel("вес. %")
plt.ylabel("Количество записей")

plt.subplot(3, 3, 8)
data['Ba'].hist(bins=30, edgecolor='black')
plt.title("Барий")
plt.xlabel("вес. %")
plt.ylabel("Количество записей")

plt.subplot(3, 3, 9)
data['Fe'].hist(bins=30, edgecolor='black')
plt.title("Железо")
plt.xlabel("вес. %")
plt.ylabel("Количество записей")

plt.figure(figsize=(10, 6))
plt.title("Boxplot для коэффициента преломления")
sns.boxplot(data=data[['RI']], orient='h')

plt.figure(figsize=(10, 6))
plt.title("Boxplot для некоторых элементов")
sns.boxplot(data=data[['Na', 'Mg', 'Al', 'K', 'Ca']], orient='h')

plt.figure(figsize=(10, 6))
sns.countplot(data=data['Type'])
plt.title("Распределение типов стекла")
plt.xlabel("Тип стекла")
plt.ylabel("Количество образцов")

sns.pairplot(data=data[['Na', 'Mg', 'Al', 'Si', 'K', 'Ca']])

data.drop(columns=["Id", "Type"], inplace=True)

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), cmap=plt.cm.Blues)

