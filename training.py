import pandas as pd
from SimpleLinearRegression import SimpleLinearRegression
from matplotlib import pyplot as plt
import numpy as np

try:
	data = pd.read_csv("data.csv")
	if (len(data['price']) != len(data['km']) or len(data["km"]) == 0):
		raise Exception("Missing rows.")
except:
	print("Missing or ill-formed data.csv file.")
	exit(-1)

print("Creating model...", end = ' ', flush = True)
linear_regressor = SimpleLinearRegression()
print("done!")

print("Training model...", end = ' ', flush = True)
x = data["km"]
y = data["price"]

linear_regressor.fit(x, y)
print("done!")

theta_0, theta_1 = linear_regressor.theta_0, linear_regressor.theta_1
print(f"theta_0 = {theta_0}, theta_1 = {theta_1}, with an R2 score = {linear_regressor.r2_score}")

print("Plotting data...", end = ' ', flush = True)
x_line = np.array([min(x), max(x)])
y_line = x_line * theta_1 + theta_0

plt.plot(x, y, 'bx')
plt.plot(x_line, y_line, 'r')
plt.xlabel("mileage (km)")
plt.ylabel("price")
plt.show(block=True)
print("done!")

print("Saving coefficients...", end = ' ', flush = True)
linear_regressor.export_coef()
print("done!")

exit(0)
