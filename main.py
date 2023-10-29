import numpy as np
import matplotlib.pyplot as plt


class KMeansClustering:
    def __init__(self, k):
        self.k = k
        self.centroids = None
        self.inertia = 0
        self.cluster_points_distances = {}
    
    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def fit(self, X, max_iterations=200):
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                                           size=(self.k, X.shape[1]))
        

        for _ in range(max_iterations):
            y = []

            self.cluster_points_distances = {}
            for data_point in X:
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)

                if not cluster_num in self.cluster_points_distances:
                    self.cluster_points_distances[cluster_num] = []
                else:
                    self.cluster_points_distances[cluster_num].append(distances[cluster_num])

                y.append(cluster_num)
            
            y = np.array(y)
            
            cluster_indices = []
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))
            
            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)
        
        for key in self.cluster_points_distances:
            self.inertia += sum(map(lambda x: x**2, self.cluster_points_distances[key]))

        return y

random_points = np.random.randint(5, 10, (200, 4))

K = range(2,20)
inertias = []

for k in K:
    kmeans = KMeansClustering(k)
    labels = kmeans.fit(random_points)
    inertias.append(kmeans.inertia)

    print(f'{k} кластеров')
    print('Центры кластеров:')
    print(kmeans.centroids)
    print('Расстояние от центра кластера до своих точек:')
    print(kmeans.cluster_points_distances)
    print('----------')

    # plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
    # plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)),
    #             marker="*", s=200)
    # plt.show()

plt.plot(K, inertias, 'bx-')
plt.xlabel('K')
plt.ylabel('Ошибка')
plt.title('График убывания ошибки')
plt.show()