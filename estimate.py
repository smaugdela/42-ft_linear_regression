import pandas

def linear_regression(theta0, theta1, mileage):
	return ((theta1 * mileage) + theta0)

if __name__ == "__main__":
	try:
		dataframe_parameters = pandas.read_csv("parameters.csv")
		dataframe_parameters.set_index("parameter", inplace=True)

		theta0 = dataframe_parameters.loc["theta0", "value"]
		theta1 = dataframe_parameters.loc["theta1", "value"]

	except:
		print("parameters.csv missing or ill-formed.")
		exit(-1)

	mileage = input("What is your car mileage? : ")
	if (mileage.isdigit() and int(mileage) >= 0):
		print(f"Estimated car price: {linear_regression(theta0, theta1, int(mileage))}")
	else:
		print("Incorrect input.")
	exit(0)
