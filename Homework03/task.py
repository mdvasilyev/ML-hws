import numpy as np

# Task 1

def mse(y_true:np.ndarray, y_predicted:np.ndarray):
    return np.mean((y_true - y_predicted) ** 2)

def r2(y_true:np.ndarray, y_predicted:np.ndarray):
    mean_observed = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_observed) ** 2)
    ss_res = np.sum((y_true - y_predicted) ** 2)
    r_sq = 1 - (ss_res / ss_tot)
    return r_sq

# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None # Save weights here
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        X_b = np.c_[np.ones((X.shape[0], 1)), X] # free variable
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_predicted = X_b.dot(self.weights)
        return y_predicted
    
# Task 3

class GradientLR:
    def __init__(self, alpha:float, iterations=10000, l=0.):
        self.alpha = alpha
        self.iterations = iterations
        self.l = l
        self.weights = None # Save weights here
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        X_b = np.c_[np.ones((X.shape[0], 1)), X] # free variable
        num_samples, num_features = X_b.shape
        # self.weights = np.random.randn(num_features)
        self.weights = np.zeros(num_features)
        for _ in range(self.iterations):
            y_pred = X_b.dot(self.weights)
            gradient = X_b.T.dot(y_pred - y) / num_samples + self.l * np.sign(self.weights)
            self.weights -= self.alpha * gradient

    def predict(self, X:np.ndarray):
        X_b = np.c_[np.ones((X.shape[0], 1)), X] # free variable
        y_predicted = X_b.dot(self.weights)
        return y_predicted

# Task 4

def get_feature_importance(linear_regression):
    weights = linear_regression.weights[1:]
    return np.abs(weights)

def get_most_important_features(linear_regression):
    feature_importance = get_feature_importance(linear_regression)
    return np.argsort(feature_importance)[::-1]