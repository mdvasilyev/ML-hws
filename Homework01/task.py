import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas
from typing import NoReturn, Tuple, List
import heapq

# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.
    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).
    """
    df = pandas.read_csv(path_to_csv).sample(frac=1).reset_index(drop=True)
    df.replace({'B': 0, 'M': 1}, inplace=True)
    return (np.array(df.loc[:, df.columns != 'label']), np.array(df['label']))

def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.
    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    """
    df = pandas.read_csv(path_to_csv).sample(frac=1).reset_index(drop=True)
    return (np.array(df.loc[:, df.columns != 'label']), np.array(df['label']))

# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float = 0.9) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.
    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.
    """
    X_train, X_test = np.array_split(X, [int(X.shape[0]*ratio)], axis=0)
    y_train, y_test = np.split(y, [int(len(y)*ratio)])
    return (X_train, y_train, X_test, y_test)

# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """
    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.
    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).
    """
    classes = len(np.unique(list(y_pred) + list(y_true)))
    precision = [[] for _ in range(classes)]
    recall = [[] for _ in range(classes)]
    def calculate_metrics(y_true, y_pred, label):
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        return precision, recall
    for i in range(classes):
        precision[i], recall[i] = calculate_metrics(y_true, y_pred, i)
    accuracy = np.mean(y_true == y_pred)
    return (precision, recall, accuracy)

# Task 4

class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        """
        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области, 
            в которых не меньше leaf_size точек).
        Returns
        -------
        """

        # Euclidian distance
        def calculate_distance(x, y):
            return np.sqrt(np.sum([(x[i]-y[i])**2 for i in range(np.size(x))]))

        # Create KDTree
        def build(X, axis=0):
            if len(X) >= leaf_size:
                X = sorted(X, key=lambda x: x[axis])
                axis = (axis + 1) % dim
                m = len(X) // 2
                return [build(X[:m], axis), build(X[m + 1:], axis), 
                    X[m]]
            if len(X) < leaf_size:
                return [None, None, X]

        # Get k nearest neighbors    
        def query(node, point, k, return_distance, heap, axis=0, decide=1):
            if node is not None:
                cur_root = node[2]
                if np.shape(cur_root) == (dim,):
                    distance = calculate_distance(point, cur_root)
                    edge_dist = cur_root[axis] - point[axis]
                    if len(heap) < k:
                        heapq.heappush(heap, (-distance, decide, cur_root))
                    elif distance < -heap[0][0]:
                        heapq.heappushpop(heap, (-distance, decide, cur_root))
                    axis = (axis + 1) % dim
                    if edge_dist < -heap[0][0]:
                        choose = (edge_dist < 0, edge_dist >= 0)[:2]
                    else:
                        choose = (edge_dist < 0, edge_dist >= 0)[:1]
                    for i in choose:
                        query(node[i], point, k, return_distance, heap, axis, decide * 2 | i)
                else:
                    for p in cur_root:
                        distance = calculate_distance(point, p)
                        edge_dist = p[axis] - point[axis]
                        if len(heap) < k:
                            heapq.heappush(heap, (-distance, decide, p))
                        elif distance < -heap[0][0]:
                            heapq.heappushpop(heap, (-distance, decide, p))
                        axis = (axis + 1) % dim
                        if edge_dist < -heap[0][0]:
                            choose = (edge_dist < 0, edge_dist >= 0)[:2]
                        else:
                            choose = (edge_dist < 0, edge_dist >= 0)[:1]
                        for i in choose:
                            query(node[i], point, k, return_distance, heap, axis, decide * 2 | i)
            if decide == 1:
                return [(-h[0], np.where(X == h[2])[0][0]) if return_distance else np.where(X == h[2])[0][0] for h in sorted(heap)][::-1]

        dim = np.shape(X)[1]
        self._query = query 
        self._root = build(X)

    
    def query(self, X: np.array, k: int = 1, return_distance=False) -> List[List]:
        """
        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.
        Returns
        -------
        list[list]
            Список списков (длина каждого списка k): 
            индексы k ближайших соседей для всех точек из X.
        """
        return [self._query(self._root, point, k, return_distance, []) for point in X]
        
        
# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """        
    
    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """        
        
    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """
    
        
    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        np.array
            Вектор предсказанных классов.
            

        """
        return np.argmax(self.predict_proba(X), axis=1)
