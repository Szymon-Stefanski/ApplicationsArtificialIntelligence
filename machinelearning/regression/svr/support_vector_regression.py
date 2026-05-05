import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('../Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


y = y.reshape(len(y), 1)


sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y)


regressor = SVR(kernel='rbf')
regressor.fit(X_scaled, y_scaled.ravel())


val_scaled = sc_X.transform([[6.5]])
pred_scaled = regressor.predict(val_scaled)

y_pred = sc_y.inverse_transform(pred_scaled.reshape(-1, 1))

print(f"Przewidziana pensja dla 6.5: {y_pred[0][0]} USD")


X_grid = np.arange(np.min(X), np.max(X), 0.1).reshape(-1, 1)
y_grid = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, y_grid, color='blue')
plt.title('Support Vector Regression (SVR)')
plt.xlabel('Poziom')
plt.ylabel('Pensja')
plt.show()
