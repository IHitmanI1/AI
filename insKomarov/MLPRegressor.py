###################### Задание 6 ###################
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
x_train = np.arange(0.0, 1, 0.001).reshape(-1, 1)
y_train = np.sin(2 * np.pi * x_train).ravel()
nn = MLPRegressor(hidden_layer_sizes=(3),
                  activation='tanh', solver='lbfgs')
n = nn.fit(x_train, y_train)
x_test = np.arange(0.0, 1, 0.05).reshape(-1, 1)
y_test = nn.predict(x_test)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x_train, y_train, s=1, c='b', marker="s", label='real')#СИНУСОИДА - Обучающие данные
ax1.scatter(x_test,  y_test,  s=10, c='r', marker="o", label='NN Prediction') #ТОЧКИ - ПРЕДСКАЗАНИЯ
plt.show()

###################### Задание 7 ####################
from time import time
import pandas as pd
import matplotlib.pyplot as plt
#данные
#Файл содержит все данные. В частности, он содержит (ДАННЫЕ МАСШТАБИРОВАНЫ!!!!)
# параметры:медианную стоимость дома, средний доход, средний возраст жилья,
# общее количество комнат, общее количество спален, население, домохозяйства,
# широту и долготу в указанном порядке
# От этих параметров зависит стоимость - целевая функция - TARGET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.inspection import plot_partial_dependence
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing  # Данные
#++++++++++++++++++++++
cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names) #Параметры, от которых зависит стоимость
y = cal_housing.target  # СТОИМОСТИ ДОМОВ - целевая переменая
print(X.iloc[:10, 0:6])
print(X.iloc[:10, 6:8])
print(y[:10])
y -= y.mean()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=0)
#+++++++++++++++++++++++++++++++++
print("Training MLPRegressor...")
tic = time()
est = make_pipeline(QuantileTransformer(),
                    MLPRegressor(hidden_layer_sizes=(50, 50),
                                 learning_rate_init=0.01,
                                 early_stopping=True))
est.fit(X_train, y_train)
print("done in {:.3f}s".format(time() - tic))
print("Test R2 score: {:.2f}".format(est.score(X_test, y_test)))
#+++++++++++++++++++++++++++++++
print('Computing partial dependence plots...')
tic = time()
# We don't compute the 2-way PDP (5, 1) here, because it is a lot slower
# with the brute method.
features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms']
plot_partial_dependence(est, X_train, features,
                        n_jobs=3, grid_resolution=20)
print("done in {:.3f}s".format(time() - tic))
fig = plt.gcf()

fig.suptitle('Зависимость стоимости дома от параметров.\n' 
             'Используется ИНС MLPRegressor')
fig.subplots_adjust(hspace=0.3)
plt.show()

