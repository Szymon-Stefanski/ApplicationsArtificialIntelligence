import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


dataset = pd.read_csv('../Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X, y)


y_pred = regressor.predict([[6.5]])
print(f"Przewidziana pensja dla poziomu 6.5: {y_pred[0]} USD")


X_grid = np.arange(np.min(X), np.max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))


plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Dane rzeczywiste')
plt.plot(X_grid, regressor.predict(X_grid), color='green', label='Random Forest (100 drzew)')
plt.title('Random Forest Regression: Poziom vs Pensja')
plt.xlabel('Poziom stanowiska')
plt.ylabel('Wynagrodzenie')
plt.legend()
plt.show()
