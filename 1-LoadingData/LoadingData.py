from sklearn import datasets

house_prices = datasets.load_boston()

print("=======================")
print("Output of Home Prices Data")
print("=======================")
print(house_prices.data)

print("=======================")
print("Output of Predicted Home Prices")
print("=======================")
print(house_prices.target)

digits = datasets.load_digits()

print("=======================")
print("Output of scikit-learn Array of Images")
print("=======================")
print(digits.images[4])
