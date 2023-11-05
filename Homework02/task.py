from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import random
import copy
import cv2
from collections import deque
from typing import NoReturn

# Task 1

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random", 
                 max_iter: int = 300):
        """
        
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        
    def fit(self, X: np.array, y = None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.
        
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать 
            параметры X и y, даже если y не используется).
        
        """
        # choose cluster init
        if self.init == "random":
            self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.init == "sample":
            self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.init == "k-means++":
            self.centroids = self.kplusplus_init(X)

        for _ in range(self.max_iter):
            # detect clusters
            labels = self.predict(X)

            # refresh centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # stop if centroids don't change
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids
    
    def kplusplus_init(self, X):
        centroids = [X[np.random.choice(X.shape[0])]
                     ]
        for _ in range(1, self.n_clusters):
            # find closest centroid for every point
            distances = np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2)
            min_distances = np.min(distances, axis=1)
            # choose next centroid ~ distance to closest centroid
            new_centroid = X[np.random.choice(X.shape[0], p=min_distances / np.sum(min_distances))]
            centroids.append(new_centroid)
        return np.array(centroids)
    
    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера, 
        к которому относится данный элемент.
        
        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.
        
        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров 
            (по одному индексу для каждого элемента из X).
        
        """
        # choose closest centroid for every point
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    
# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 leaf_size: int = 40, metric: str = "euclidean"):
        """
        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть 
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean 
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric
        self.labels = None
        
    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).
        """
        n = X.shape[0]
        self.labels = np.full(n, -1)
        cluster_label = 0
        tree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        kdtree = tree.query_radius(X, r=self.eps).tolist()
        for i in range(n):
            # skip if belongs to cluster
            if self.labels[i] != -1:
                continue
            neighbors = kdtree[i]
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                self.labels[i] = cluster_label
                closest = [i]
                while closest:
                    idx = closest.pop()
                    neighbors = kdtree[idx]
                    for neighbor in neighbors:
                        if self.labels[neighbor] == -1:
                            self.labels[neighbor] == cluster_label
                            if len(kdtree[neighbor]) >= self.min_samples:
                                closest.append(neighbor)
                cluster_label += 1
        return self.labels
    
# Task 3

class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """
        
        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры 
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        pass
        # self.n_clusters = n_clusters
        # self.linkage = linkage
    
    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        pass
    #     n = X.shape[0]
    #     self.labels = np.arange(n)  # Инициализируем каждую точку как отдельный кластер

    #     while len(np.unique(self.labels)) > self.n_clusters:
    #         min_distance = np.inf
    #         cluster1, cluster2 = -1, -1

    #         for i in range(n):
    #             for j in range(i + 1, n):
    #                 if self.labels[i] == self.labels[j]:
    #                     continue
    #                 distance = self.calculate_distance(X, i, j)
    #                 if distance < min_distance:
    #                     min_distance = distance
    #                     cluster1, cluster2 = self.labels[i], self.labels[j]

    #         # Объединяем два кластера с наименьшим расстоянием
    #         self.labels[self.labels == cluster2] = cluster1

    #     return self.labels
    
    # def calculate_distance(self, X, i, j):
    #     if self.linkage == "average":
    #         cluster_i = X[self.labels == self.labels[i]]
    #         cluster_j = X[self.labels == self.labels[j]]
    #         return np.mean(np.linalg.norm(cluster_i[:, np.newaxis] - cluster_j, axis=2))
    #     elif self.linkage == "single":
    #         cluster_i = X[self.labels == self.labels[i]]
    #         cluster_j = X[self.labels == self.labels[j]]
    #         return np.min(np.linalg.norm(cluster_i[:, np.newaxis] - cluster_j, axis=2))
    #     elif self.linkage == "complete":
    #         cluster_i = X[self.labels == self.labels[i]]
    #         cluster_j = X[self.labels == self.labels[j]]
    #         return np.max(np.linalg.norm(cluster_i[:, np.newaxis] - cluster_j, axis=2))
