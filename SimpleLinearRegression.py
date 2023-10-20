import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

class SimpleLinearRegression:

	def __init__(self) -> None:
		self.theta_0 = 0.0
		self.theta_1 = 0.0
		self.r2_score = 0.0

	def fit(self, x: np.array, y: np.array, learning_rate=0.1, iterations=10000, precision=1e-6, visualizer=False):
		"""
		Finds the best parameters theta_0 and theta_1 (bias and weight) for given input X and output y, using gradient descent.
		Another way could be to estimate the coefficients using the pricniple of least squares.
		"""

		if len(x) != len(y):
			raise ValueError("The dataset is invalid. Their length differ")

		norm_theta_0 = 0.0
		norm_theta_1 = 0.0

		# Normalizing data
		max_x = max(x)
		min_x = min(x)
		x = (x - min_x) / (max_x - min_x)
		max_y = max(y)
		min_y = min(y)
		y = (y - min_y) / (max_y - min_y)

		m = float(len(x))

		if visualizer:
			# Setup
			x_line = np.array([min(x), max(x)])
			y_line = x_line * norm_theta_1 + norm_theta_0
			plt.ion()
			fig = plt.figure()
			ax = fig.add_subplot(111) 
			ax.plot(x, y, 'bx')
			line, = ax.plot(x_line, y_line, 'r')

		for i in range(iterations):

			if visualizer and plt.fignum_exists(fig.number):
				y_line = x_line * norm_theta_1 + norm_theta_0
				line.set_ydata(y_line)
				fig.canvas.draw()
				fig.canvas.flush_events()

			y_pred = norm_theta_0 + (norm_theta_1 * x)
			
			tmp_theta_0 = learning_rate / (2 * m) * np.sum(y_pred - y)
			tmp_theta_1 = learning_rate / (2 * m) * np.sum((y_pred - y) * x)

			norm_theta_0 = norm_theta_0 - tmp_theta_0
			norm_theta_1 = norm_theta_1 - tmp_theta_1

			if (abs(tmp_theta_0) < precision and abs(tmp_theta_1) < precision):
				break

		# Computing R2 score
		y_mean = y.mean()
		y_pred = norm_theta_0 + (norm_theta_1 * x)
		self.r2_score =  1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2))

		# Denormalizing final parameters
		self.theta_0 = norm_theta_0 * (max_y - min_y) + min_y
		self.theta_1 = norm_theta_1 * (max_y - min_y) / (max_x - min_x)

	def predict(self, x):
		return self.theta_0 + (self.theta_1 * x)

	def import_coef(self, file_name: str = 'simple_regressor.pkl'):
		try:
			with open(file_name, 'rb') as file:
				self.theta_0 = pickle.load(file)
				self.theta_1 = pickle.load(file)
				self.r2_score = pickle.load(file)
		except:
			print("No file found.")

	def export_coef(self, file_name: str = 'simple_regressor.pkl'):
		try:
			with open(file_name, 'wb') as file:
				pickle.dump(self.theta_0, file)
				pickle.dump(self.theta_1, file)
				pickle.dump(self.r2_score, file)
		except:
			print("Could not export to file.")
