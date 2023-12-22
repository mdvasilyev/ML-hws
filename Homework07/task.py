from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 1

def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    unique, counts = np.unique(x, return_counts=True)
    prob = counts / len(x)
    gini = 1.0 - np.sum(prob ** 2)
    return gini

def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    unique, counts = np.unique(x, return_counts=True)
    prob = counts / len(x)
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    length = len(left_y) + len(right_y)
    metric = criterion(np.concatenate([left_y, right_y]))
    l_weight = len(left_y) / length
    r_weight = len(right_y) / length
    gain = metric - (l_weight * criterion(left_y) + r_weight * criterion(right_y))
    return gain


# Task 2

class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """
    def __init__(self, ys):
        unique, counts = np.unique(ys, return_counts=True)
        self.y = unique[np.argmax(counts)]

class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value. 
    """
    def __init__(self, split_dim: int, split_value: float, 
                 left: Union['DecisionTreeNode', DecisionTreeLeaf], 
                 right: Union['DecisionTreeNode', DecisionTreeLeaf],
                 info_gain: float = {}):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
# Task 3

class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """
    def __init__(self, criterion : str = "gini", 
                 max_depth : Optional[int] = None, 
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def build(self, dataset: np.ndarray, cur_depth: int):
        X, y = dataset[:, :-1], dataset[:, -1]
        samples, features = np.shape(X)
        if self.max_depth is None: self.max_depth = 10e6
        if samples >= self.min_samples_leaf and cur_depth <= self.max_depth:
            split = self.get_split(dataset, features)
            if split["info_gain"] > 0:
                left = self.build(split["left"], cur_depth + 1)
                right = self.build(split["right"], cur_depth + 1)
                return DecisionTreeNode(split["split_dim"], split["split_value"], left, right, split["info_gain"])
        return DecisionTreeLeaf(y)
    
    def get_split(self, dataset: np.ndarray, features: int):
        split = {}
        max_info_gain = -10e6
        for split_dim in range(features):
            feature_values = dataset[:, split_dim]
            possible_thresholds = np.unique(feature_values)
            for split_value in possible_thresholds:
                left_dataset = np.array([row for row in dataset if row[split_dim] < split_value])
                right_dataset = np.array([row for row in dataset if row[split_dim] >= split_value])
                if len(left_dataset) > 0 and len(right_dataset) > 0:
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    cur_info_gain = gain(left_y, right_y, globals()[self.criterion])
                    if cur_info_gain > max_info_gain:
                        max_info_gain = cur_info_gain
                        split["split_dim"] = split_dim
                        split["split_value"] = split_value
                        split["left"] = left_dataset
                        split["right"] = right_dataset
                        split["info_gain"] = cur_info_gain
        return split

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        dataset = np.hstack((X, np.transpose([y])))
        self.root = self.build(dataset, cur_depth=0)
    
    def predict_proba(self, X: np.ndarray) ->  List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь 
            {метка класса -> вероятность класса}.
        """
        proba = {}
        for x in X:
            proba[x] = self.traverse_tree(x, self.root)
        return proba
    
    def traverse_tree(self, x, tree: Union[DecisionTreeNode, DecisionTreeLeaf]):
        if tree.__class__.__name__ == "DecisionTreeLeaf":
            return tree.y
        value = x[tree.split_dim]
        if value < tree.split_value:
            return self.traverse_tree(x, tree.left)
        else:
            return self.traverse_tree(x, tree.right)
    
    def predict(self, X : np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]
    
# Task 4
task4_dtc = None

