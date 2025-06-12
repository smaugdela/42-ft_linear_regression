from SimpleLinearRegression import SimpleLinearRegression


def main():
	linear_regressor = SimpleLinearRegression()
	linear_regressor.import_coef()

	mileage = input("What is your car mileage? : ")
	if (mileage.isdigit() and int(mileage) >= 0):
		print(f"Estimated car price: {linear_regressor.predict(int(mileage))}")
		return 0
	else:
		print("Incorrect input.")
		return 42


if __name__ == "__main__":
	try:
		exit(main())
	except Exception as e:
		print(f"Error: {e}")
		exit(42)
