import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


dataset = pd.read_csv('../Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values


lin_reg = LinearRegression()
lin_reg.fit(X, y)


poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)


lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue', label='Liniowa')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='green', label='Wielomianowa')
plt.show()
