import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


data = pd.read_csv('Wholesale_customers_data.csv')

print(data.head())
print("\n----------------------------------------------------------------------")

X = data.drop(['Channel', 'Region'], axis=1)

print(X.head())
print("\n----------------------------------------------------------------------")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nПромасштабированные признаки:")
print(X_scaled[:5])
print("\n----------------------------------------------------------------------")

# --- 

k_range = range(1, 11)
wcss = []

for i in k_range:
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o')
plt.xlabel('Количество кластеров k')
plt.ylabel('WCSS')
plt.title('Метод локтя')
plt.grid()

# --- 

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_scaled)

data['y_kmeans'] = y_kmeans

# ---

X_vis = X_scaled[:, [0, 2]]

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X_vis[y_kmeans == i, 0], X_vis[y_kmeans == i, 1], 
                c=colors[i], label=f'Cluster {i+1}', alpha=0.6)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 2], 
            s=100, c='yellow', edgecolors='black', label='Centroids')

plt.title('Кластеры потребителей')
plt.xlabel('Fresh (Свежие продукты)')
plt.ylabel('Grocery (Бакалея)')
plt.legend()

# ---

X_vis = X_scaled[:, [3, 2]]

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X_vis[y_kmeans == i, 0], X_vis[y_kmeans == i, 1], 
                c=colors[i], label=f'Cluster {i+1}', alpha=0.6)

plt.scatter(kmeans.cluster_centers_[:, 3], kmeans.cluster_centers_[:, 2], 
            s=100, c='yellow', edgecolors='black', label='Centroids')

plt.title('Кластеры потребителей')
plt.xlabel('Frozen (Замороженные продукты)')
plt.ylabel('Grocery (Бакалея)')
plt.legend()

# ---

X_vis = X_scaled[:, [0, 4]]

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X_vis[y_kmeans == i, 0], X_vis[y_kmeans == i, 1], 
                c=colors[i], label=f'Cluster {i+1}', alpha=0.6)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 4], 
            s=100, c='yellow', edgecolors='black', label='Centroids')

plt.title('Кластеры потребителей')
plt.xlabel('Fresh (Свежие продукты)')
plt.ylabel('Detergents Paper (Бытовая химия и бумажные товары)')
plt.legend()

# ---

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_pca)

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], 
                c=colors[i], label=f'Cluster {i+1}', alpha=0.6)
    
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=100, c='yellow', edgecolors='black', label='Centroids')

plt.title('Кластеры потребителей (PCA)')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.legend()