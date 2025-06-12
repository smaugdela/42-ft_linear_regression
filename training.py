import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser, Namespace

from SimpleLinearRegression import SimpleLinearRegression


def main(args: Namespace):

	data = pd.read_csv(args.file)
	if (len(data['price']) != len(data['km']) or len(data["km"]) == 0):
		raise ValueError("Invalid data in data.csv file.")

	print("Creating model...", end = ' ', flush = True)
	linear_regressor = SimpleLinearRegression()
	print("done!")

	print("Training model...", end = ' ', flush = True)
	x = np.array(data["km"])
	y = np.array(data["price"])

	linear_regressor.fit(x, y, visualizer=args.visualizer)
	print("done!")

	theta_0, theta_1, r2 = linear_regressor.theta_0, linear_regressor.theta_1, linear_regressor.r2_score
	print(f"theta_0 = {theta_0}, theta_1 = {theta_1}" + (f", with an R2 score = {r2}" if args.bonus else ""))

	if args.bonus:
		print("Plotting data...", end = ' ', flush = True)
		x_line = np.array([min(x), max(x)])
		y_line = x_line * theta_1 + theta_0

		plt.plot(x, y, 'bx')
		plt.plot(x_line, y_line, 'r')
		plt.xlabel("mileage (km)")
		plt.ylabel("price")
		plt.title(f"Theta_0 = {theta_0}, Theta_1 = {theta_1}, R2 = {r2}")
		plt.show(block=True)
		print("done!")

	print("Saving coefficients...", end = ' ', flush = True)
	linear_regressor.export_coef()
	print("done!")

	return 0


if __name__ == "__main__":

	parser = ArgumentParser(description="Train a simple linear regression model on car mileage and price data.")
	parser.add_argument("file", type=str, help="Path to the CSV file containing the data.")
	parser.add_argument("--bonus", "-b", action="store_true", help="Enable bonus features (plotting and R2 score).", default=False)
	parser.add_argument("--visualizer", "-v", action="store_true", help="Enable visualizer for training process.", default=False)
	args = parser.parse_args()

	try:
		exit(main(args))
	except Exception as e:
		print(f"Error: {e}")
		exit(42)
