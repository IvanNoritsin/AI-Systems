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
plt.subplot(3, 3, 1)
data['RI'].hist(bins=30, edgecolor='black')
plt.title("Коэффициент преломления")

plt.subplot(3, 3, 2)
data['Na'].hist(bins=30, edgecolor='black')
plt.title("Натрий")

plt.subplot(3, 3, 3)
data['Mg'].hist(bins=30, edgecolor='black')
plt.title("Магний")

plt.subplot(3, 3, 4)
data['Al'].hist(bins=30, edgecolor='black')
plt.title("Алюминий")

plt.subplot(3, 3, 5)
data['Si'].hist(bins=30, edgecolor='black')
plt.title("Кремний")

plt.subplot(3, 3, 6)
data['K'].hist(bins=30, edgecolor='black')
plt.title("Калий")

plt.subplot(3, 3, 7)
data['Ca'].hist(bins=30, edgecolor='black')
plt.title("Кальций")

plt.subplot(3, 3, 8)
data['Ba'].hist(bins=30, edgecolor='black')
plt.title("Барий")

plt.subplot(3, 3, 9)
data['Fe'].hist(bins=30, edgecolor='black')
plt.title("Железо")

plt.figure(figsize=(10, 6))
plt.title("Boxplot для коэффициента преломления")
sns.boxplot(data=data[['RI']], orient='h')

plt.figure(figsize=(10, 6))
plt.title("Boxplot для некоторых элементов")
sns.boxplot(data=data[['Na', 'Mg', 'Al', 'K', 'Ca']], orient='h')

sns.pairplot(data=data[['Na', 'Mg', 'Al', 'Si', 'K', 'Ca']])

data.drop(columns=["Id", "Type"], inplace=True)

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), cmap=plt.cm.Blues)

