import pandas
from SimpleLinearRegression import SimpleLinearRegression

linear_regressor = SimpleLinearRegression()

linear_regressor.import_coef()

mileage = input("What is your car mileage? : ")
if (mileage.isdigit() and int(mileage) >= 0):
	print(f"Estimated car price: {linear_regressor.predict(int(mileage))}")
else:
	print("Incorrect input.")
exit(0)
