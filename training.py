import pandas
import matplotlib.pyplot as plt

def	get_estimate(mileage, theta0, theta1):
	"""
	Return estimate value of price given mileage, theta0 and theta1
	"""
	return ((theta1 * mileage) + theta0)

def	cost_function():
	"""
	Compute the cost function of a given x, y list for fixed theta0, theta1.
	"""
	pass

def gradient_descent(learning_rate: float, mileage_list, price_list, max_iter, epsilon):
	"""
	Main logic for learning algorithm given learning_rate (alpha) and returns a tuple of the computed theta0 and theta1.
	max_iter is the maximum number of iteration that will be run, to avoid infinite loop in case of divergence.
	epsilon is the threshold under which the gradient descent will consider to be precise enough and stop there.
	"""

	theta0 = theta1 = temp0 = temp1 = 0

	m = len(mileage_list)
	coef = learning_rate / m

	i = -1
	while (++i < max_iter):

		error = 0
		for (mileage, price) in zip(mileage_list, price_list):
			error += (get_estimate(mileage, theta0, theta1) - price)

		temp0 = coef * error

		error = 0
		for (mileage, price) in zip(mileage_list, price_list):
			error += (get_estimate(mileage, theta0, theta1) - price) * mileage

		temp1 = coef * error

		theta0 = theta0 - temp0
		theta1 = theta1 - temp1

		if (temp0 < epsilon and temp1 < epsilon):
			break

	return(theta0, theta1)

if __name__ == "__main__":
	try:
		data = pandas.read_csv("data.csv")
		if (len(data['price']) != len(data['km'])):
			raise Exception("Bad csv.")

	except:
		print("Missing or ill-formed data.csv file.")
		exit(-1)

	print("Training...")

	new_parameters = gradient_descent(0.01, data['km'], data['price'], 10000000, 0.00001)

	fig = plt.subplots()
	fig.scatter(data['km'], data['price'])
	fig.plot()

	try:
		data = {
			'parameter': ['theta0', 'theta1'],
			'value': [new_parameters[0], new_parameters[1]]
		}
		df = pandas.DataFrame(data)
		df.set_index("parameter", inplace=True)
		df.to_csv("parameters.csv")

	except:
		print("Could not write to parameters.csv file.")
		exit(-1)

	print("done!")
	exit(0)