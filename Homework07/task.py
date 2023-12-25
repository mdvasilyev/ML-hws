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
        self.prob = dict(zip(unique, counts))

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
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        
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
        if criterion == "gini":
            self.criterion = gini
        else:
            self.criterion = entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def build(self, X: np.ndarray, y: np.ndarray, cur_depth: int):
        unique_ys = set(y)
        if (self.max_depth is not None and cur_depth >= self.max_depth) or (len(y) <= self.min_samples_leaf or len(unique_ys) == 1):
            return DecisionTreeLeaf(y)
        split_dim, split_value = self.get_split(X, y)
        if split_dim is None:
            return DecisionTreeLeaf(y)
            
        mask = X[:, split_dim] < split_value
        return DecisionTreeNode(split_dim, split_value, self.build(X[mask], y[mask], cur_depth + 1), self.build(X[np.invert(mask)], y[np.invert(mask)], cur_depth + 1))

    def calculate_gain(self, left_y: np.ndarray, right_y: np.ndarray, criterion: Callable, criterion_value, labels: np.ndarray):
        left_w = len(left_y) / len(labels)
        right_w = len(right_y) / len(labels)
        return criterion_value - (left_w * criterion(left_y) + right_w * criterion(right_y))
    
    def get_split(self, X: np.ndarray, y: np.ndarray):
        gain = -10e6
        split_dim = None
        split_value = None
        total_entropy = self.criterion(y)
        for dim in range(X.shape[1]):
            values = np.unique(X[:, dim])
            if len(values) != 1:
                for value in values:            
                    mask = X[:, dim] < value
                    if (len(y) - len(y[mask]) >= self.min_samples_leaf) or len(y[mask]) >= self.min_samples_leaf:
                        cur_gain = self.calculate_gain(y[mask], y[np.invert(mask)], self.criterion, total_entropy, y)
                        if cur_gain > gain:
                            gain = cur_gain
                            split_dim = dim
                            split_value = value
        return split_dim, split_value

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
        self.root = self.build(X, y, cur_depth=0)
    
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
        proba = [self.predict_class(self.root, x) for x in X]
        return proba
    
    def predict_class(self, node, x):
        if isinstance(node, DecisionTreeLeaf):
            return node.prob
        elif x[node.split_dim] < node.split_value:
            return self.predict_class(node.left, x)
        return self.predict_class(node.right, x)
    
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
task4_dtc = DecisionTreeClassifier("gini", 5, 2)

