import pandas
import matplotlib.pyplot as plt
import numpy as np

# Least squares method
def estimate_coef(x, y):
	n = np.size(x)
	m_x, m_y = np.mean(x), np.mean(y)

	SS_xy = np.sum(y*x) - n*m_y*m_x
	SS_xx = np.sum(x*x) - n*m_x*m_x

	theta_1 = SS_xy / SS_xx
	theta_0 = m_y - theta_1*m_x

	return(theta_0, theta_1)

# Gradient descent method
def gradient_descent(x: np.array, y: np.array, learning_rate=0.0001, iterations=10000, precision=1e-6):

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

		for i in range(iterations):

			y_pred = norm_theta_0 + (norm_theta_1 * x) 
			
			tmp_theta_0 = learning_rate / (2 * m) * np.sum(y_pred - y)
			tmp_theta_1 = learning_rate / (2 * m) * np.sum((y_pred - y) * x)

			norm_theta_0 = norm_theta_0 - tmp_theta_0
			norm_theta_1 = norm_theta_1 - tmp_theta_1

			if (abs(tmp_theta_0) < precision and abs(tmp_theta_1) < precision):
				break
		
		# Denormalizing final parameters
		theta_0 = norm_theta_0 * (max_y - min_y) + min_y
		theta_1 = norm_theta_1 * (max_y - min_y) / (max_x - min_x)

		return theta_0, theta_1

if __name__ == "__main__":

	try:
		data = pandas.read_csv("data.csv")
		if (len(data['price']) != len(data['km']) or len(data["km"]) == 0):
			raise Exception("Missing rows.")
	except:
		print("Missing or ill-formed data.csv file.")
		exit(-1)

	print("Training...")

	(theta0, theta1) = gradient_descent(data["km"], data["price"])
	# (theta0, theta1) = estimate_coef(data['km'], data['price'])

	print(theta0, theta1)

	min = min(data["km"])
	max = max(data["km"])
	x = np.linspace(min, max, 2)
	y = theta0 + (theta1 * x)

	plt.plot(data["km"], data["price"], 'bx')
	plt.plot(x, y, "r")
	plt.xlabel("mileage (km)")
	plt.ylabel("price")
	plt.show()

	try:
		data = {
			'parameter': ['theta0', 'theta1'],
			'value': [theta0, theta1]
		}
		df = pandas.DataFrame(data)
		df.set_index("parameter", inplace=True)
		df.to_csv("parameters.csv")
	except:
		print("Could not write to parameters.csv file.")
		exit(-1)

	print("...done!")
	exit(0)
