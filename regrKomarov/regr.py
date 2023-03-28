import pandas
import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv("Boston.csv", delimiter=",", names=names)
array = dataframe.values
X = dataframe[['LSTAT']].values
Y = dataframe['MEDV'].values
model = KNeighborsRegressor(n_neighbors = 30)
model.fit(X,Y)
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(Y,y_pred))
r2 = r2_score(Y,y_pred)
print('rmse=', rmse)
print('Точность модели - r2',r2)
#вывод графика
import matplotlib.pyplot as plt
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y,edgecolors='g')
plt.scatter(X, y_pred,edgecolors='m')
plt.plot(X_grid, model.predict(X_grid), color = 'blue')
plt.legend(['Предсказанная','Реальная'])
plt.xlabel('Степень низкого статуса населения')
plt.ylabel('Стоимость дома (тыс.долл)')
plt.title('Нелинейная регрессия KNN от Комарова Д.И.')
plt.show()