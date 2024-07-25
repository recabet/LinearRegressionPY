import numpy as np
from typing import Tuple


class my_LinearRegression:
    """
    Linear Regression model.

    :param rate: Learning rate for gradient descent.
    :type rate: float
    :param iterations: Number of iterations for gradient descent.
    :type iterations: int
    :param normalize: Whether to normalize the features and target variables.
    :type normalize: bool
    """
    
    def __init__ (self, rate: float = 0.01, iterations: int = 1000, normalize: bool = True):
        """
        Initialize the Linear Regression model.

        :param rate: Learning rate for gradient descent.
        :type rate: float
        :param iterations: Number of iterations for gradient descent.
        :type iterations: int
        :param normalize: Whether to normalize the features and target variables.
        :type normalize: bool
        """
        self.rate = rate
        self.iterations = iterations
        self.normalize = normalize
        self.w = None
        self.b = np.random.randn()
        self.X_mean = None
        self.X_std = None
        self.y_mean = 0
        self.y_std = 1
    
    def __normalize (self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize the feature matrix X and target variable y.

        :param X: Feature matrix.
        :type X: np.ndarray
        :param y: Target variable.
        :type y: np.ndarray

        :return: Tuple of normalized feature matrix and target variable.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        X_normalized = (X - self.X_mean) / self.X_std
        y_normalized = (y - self.y_mean) / self.y_std
        return X_normalized, y_normalized
    
    def fit (self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit the Linear Regression model to the training data.

        :param X: Training feature matrix.
        :type X: np.ndarray
        :param y: Training target variable.
        :type y: np.ndarray

        :return: Tuple of weights and bias.
        :rtype: Tuple[np.ndarray, float]
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 2:
            y = y.flatten()
        
        n, m = X.shape
        self.w = np.random.randn(m)
        
        if self.normalize:
            X, y = self.__normalize(X, y)
        
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.w) + self.b
            error = y_pred - y
            w_gradient = (1 / n) * np.dot(X.T, error)
            b_gradient = (1 / n) * np.sum(error)
            self.w -= self.rate * w_gradient
            self.b -= self.rate * b_gradient
        
        if self.normalize:
            self.w = self.w * (self.y_std / self.X_std)
            self.b = self.b * self.y_std + self.y_mean - np.dot(self.w, self.X_mean)
        
        return self.w, self.b
    
    def compute_cost (self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the cost (mean squared error) of the model.

        :param X: Feature matrix.
        :type X: np.ndarray
        :param y: Target variable.
        :type y: np.ndarray

        :return: Cost value.
        :rtype: float
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 2:
            y = y.flatten()
        
        m = len(y)
        pred = self.predict(X)
        cost = (1 / (2 * m)) * np.sum((pred - y) ** 2)
        return cost
    
    def predict (self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target variable for given feature matrix X.

        :param X: Feature matrix.
        :type X: np.ndarray

        :return: Predicted target variable.
        :rtype: np.ndarray
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.w) + self.b
    
    def r_squared (self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the R-squared (coefficient of determination) of the model.

        :param X: Feature matrix.
        :type X: np.ndarray
        :param y: Target variable.
        :type y: np.ndarray

        :return: R-squared value.
        :rtype: float
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2
    
    def error (self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the percentage error of the model.

        :param X: Feature matrix.
        :type X: np.ndarray
        :param y: Target variable.
        :type y: np.ndarray

        :return: Percentage error.
        :rtype: float
        """
        y_predict = self.predict(X)
        return np.sum(np.abs(y_predict - y) / y) / len(y) * 100
