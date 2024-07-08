import numpy as np

class my_LinearRegression:
	def __init__ (self, rate=0.01, iter=1000,normalize=True):
		self.rate = rate
		self.iter = iter
		self.normalize = normalize
		self.w = None
		self.b = np.random.randn()
		self.X_mean = None
		self.X_std = None
		self.y_mean = 0
		self.y_std = 1
	
	def __normalize (self, X, y):
		self.X_mean = np.mean(X, axis=0)
		self.X_std = np.std(X, axis=0)
		self.y_mean = np.mean(y)
		self.y_std = np.std(y)
		X_normalized = (X - self.X_mean) / self.X_std
		y_normalized = (y - self.y_mean) / self.y_std
		return X_normalized, y_normalized
	
	def fit (self, X, y):
		if X.ndim == 1:
			X = X.reshape(-1, 1)
		if y.ndim == 2:
			y = y.flatten()
		
		n, m = X.shape
		self.w = np.random.randn(m)
		
		if self.normalize:
			X, y = self.__normalize(X, y)
		
		for i in range(self.iter):
			y_pred = np.dot(X, self.w) + self.b
			error = y_pred - y
			w_gradient = (1 / n) * np.dot(X.T, error)
			b_gradient = (1 / n) * np.sum(error)
			self.w -= self.rate * w_gradient
			self.b -= self.rate * b_gradient
		
		self.w = self.w * (self.y_std / self.X_std)
		self.b = self.b * self.y_std + self.y_mean - np.dot(self.w, self.X_mean)
		
		return self.w, self.b
	
	def compute_cost (self, X, y):
		if X.ndim == 1:
			X = X.reshape(-1, 1)
		if y.ndim == 2:
			y = y.flatten()
		
		m = len(y)
		pred = self.predict(X)
		cost = (1 / (2 * m)) * np.sum((pred - y) ** 2)
		return cost
	
	def predict (self, X):
		if X.ndim == 1:
			X = X.reshape(-1, 1)
		return np.dot(X, self.w) + self.b
	
	def r_squared (self, X, y):
		y_pred = self.predict(X)
		ss_total = np.sum((y - np.mean(y)) ** 2)
		ss_residual = np.sum((y - y_pred) ** 2)
		r2 = 1 - (ss_residual / ss_total)
		return r2
	
	def error (self,X, y):
		y_predict=np.array([self.predict(X)])
		return np.sum(abs(y_predict- y) /y) / len(y) * 100




