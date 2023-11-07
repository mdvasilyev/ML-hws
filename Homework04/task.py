import numpy as np
from sklearn.model_selection import train_test_split
import copy
from typing import NoReturn

# Task 1

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """
        self.learning_rate = 0.001
        self.iterations = iterations
        self.w = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон. 
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        self.w = np.zeros(X.shape[1] + 1)
        X = np.c_[np.ones((X.shape[0])), X]
        y_ = np.where(y > 0, 1, 0)
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.w)
            errors = y_ - np.where(y_pred > 0, 1, 0)
            self.w += self.learning_rate * np.dot(X.T, errors)
            
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        X = np.c_[np.ones((X.shape[0])), X]
        return np.where(np.dot(X, self.w) > 0, 1, 0)
    
# Task 2

class PerceptronBest:

    def __init__(self, iterations: int = 100, learning_rate: float = 0.01, early_stopping: bool = True, tol: float = 0.001):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.tol = tol
        self.cur_w = None
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона, 
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса, 
        при которых значение accuracy было наибольшим.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        # self.cur_w = np.zeros(X.shape[1] + 1)
        self.cur_w = np.random.rand(X.shape[1] + 1) * 0.01
        self.w = np.copy(self.cur_w)
        X = np.c_[np.ones((X.shape[0])), X]
        y_ = np.where(y > 0, 1, 0)
        best_accuracy = 0
        prev_accuracy = 0
        for _ in range(self.iterations):
            y_pred = X.dot(self.cur_w)
            errors = y_ - np.where(y_pred > 0, 1, 0)
            self.cur_w += self.learning_rate * X.T.dot(errors)
            
            cur_accuracy = np.mean(np.where(X.dot(self.cur_w) > 0, 1, 0) == y_)
            if cur_accuracy >= best_accuracy:
                best_accuracy = cur_accuracy
                self.w = np.copy(self.cur_w)
            
            if self.early_stopping and abs(cur_accuracy - prev_accuracy) < self.tol:
                break
            prev_accuracy = cur_accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        X = np.c_[np.ones((X.shape[0])), X]
        return np.where(X.dot(self.w) > 0, 1, 0)

# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов:
    1) Среднее значение яркости пикселей (mean)
    2) Стандартное отклонение яркости пикселей (std deviation)
        
    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    n_images = images.shape[0]
    transformed_images = np.zeros((n_images, 2))
    for i in range(n_images):
        image = images[i]
        brightness = image.mean()
        std_deviation = image.std()
        transformed_images[i] = [brightness, std_deviation]

    return transformed_images