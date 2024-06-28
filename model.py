import numpy as np


class LinearRegression:
	def __init__ (self, rate=0.01, iter=10000):
		self.rate = rate
		self.iter = iter
		self.k = 0
		self.b = 0
		self.X_mean = 0
		self.X_std = 1
		self.y_mean = 0
		self.y_std = 1
	
	def fit (self, X, y):
		n = len(y)
		
		self.X_mean = np.mean(X)
		self.X_std = np.std(X)
		self.y_mean = np.mean(y)
		self.y_std = np.std(y)
		
		X_normalized = (X - self.X_mean) / self.X_std
		y_normalized = (y - self.y_mean) / self.y_std
		
		for i in range(self.iter):
			y_pred = self.k * X_normalized + self.b
			error = y_pred - y_normalized
			k_gradient = (1 / n) * np.sum(error * X_normalized)
			b_gradient = (1 / n) * np.sum(error)
			self.k -= self.rate * k_gradient
			self.b -= self.rate * b_gradient
			
		self.k = self.k * (self.y_std / self.X_std)
		self.b = self.b * self.y_std + self.y_mean - self.k * self.X_mean
		
		return self.k, self.b
	
	def compute_cost (self, X, y):
		m = len(y)
		pred = self.predict(X)
		cost = (1 / (2 * m)) * np.sum((pred - y) ** 2)
		return cost
	
	def predict (self, X):
		return X * self.k + self.b
	
	def r_squared (self, X, y):
		y_pred = self.predict(X)
		ss_total = np.sum((y - np.mean(y)) ** 2)
		ss_residual = np.sum((y - y_pred) ** 2)
		r2 = 1 - (ss_residual / ss_total)
		return r2
	
	def error (self, X, y):
		err = 0
		for i in zip(self.predict(X), y):
			err += (i[0] - i[1]) / len(y) / i[1]
		return err * 100

