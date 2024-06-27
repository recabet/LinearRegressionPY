import numpy as np


class LinearRegression:
	def __init__ (self, rate=0.01, iter=10):
		self.rate = rate
		self.iter = iter
		self.k = 0
		self.b = 0
	
	def fit (self, X, y):
		n = len(y)
		self.k = np.random.randn()
		self.b = np.random.randn()
		cost_history = np.zeros(self.iter)
		
		for i in range(self.iter):
			y_pred = self.k * X + self.b
			error = y_pred - y
			k_gradient = (1 / n) * np.sum(error * X)
			b_gradient = (1 / n) * np.sum(error)
			self.k -= self.rate * k_gradient
			self.b -= self.rate * b_gradient
			
			cost = self.compute_cost(X, y)
			cost_history[i] = cost
			
			# Debugging print statement
			if (i + 1) % 1 == 0:
				print(f"Iteration {i + 1}: k = {self.k}, b = {self.b}, cost = {cost}")
		
		return self.k, self.b, cost_history
	
	def compute_cost (self, X, y):
		m = len(y)
		pred = X * self.k + self.b
		cost = np.sum((pred - y) ** 2) / (2 * m)
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
		err = np.mean((self.predict(X) - y) / y) * 100
		return err