import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)


plt.scatter(X_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(X_test, y_test, color='green', label='Dane testowe')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Wynagrodzenie vs Doświadczenie')
plt.xlabel('Lata doświadczenia')
plt.ylabel('Wynagrodzenie')
plt.legend()
plt.show()
