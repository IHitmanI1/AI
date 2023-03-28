import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
col_names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
boston_df = pd.read_csv("Boston.csv", names=col_names, delimiter=",")
print(boston_df.head())
plt.figure(figsize=(10, 7))
hm = sns.heatmap(boston_df[col_names].corr(),
                 cbar=True,
                 annot=True)
plt.show()

X = boston_df[['LSTAT']].values
Y = boston_df['MEDV'].values
#Определение модели============================
#Мы создаем экземпляр LinearRegression и называем его моделью. Затем мы подгоняем наши данные к модели
model = LinearRegression ()
model.fit(X, Y)
#Точность модели======================================
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(Y,y_pred))
r2 = r2_score(Y,y_pred)
print('rmse=', rmse)
print('Точность модели - r2 =',r2)
#вывод графика
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y,edgecolors='g')
plt.plot(X_grid, model.predict(X_grid), color = 'blue')
plt.legend(['Предсказанная','Реальная'])
plt.xlabel('Степень низкого статуса населения')
plt.ylabel('Стоимость дома (тыс.долл)')
plt.title('Базовая линейная регрессия  от Комарова Д.И.')
plt.show()

######################################### Задание 2 #########################################3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
col_names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
boston_df = pd.read_csv("Boston.csv", names=col_names, delimiter=",")
print(boston_df.head())
array = boston_df.values
X = array[:,0:13] #все строки, столбцы 0,1,2 - до 13
Y = array[:,13]  # все строки 13 столбца
#Определение модели============================
model = LinearRegression ()
model.fit (X, Y)
#форма модели: Y = a*X + b*Z + c*V +  ... + k   или Y = a0 + a1*X1 + a2*X2 + a3*X3 + ... ak*Xk
#Точность  модели======================================
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(Y,y_pred))
r2 = r2_score(Y,y_pred)
print('rmse=', rmse)
print('Точность модели - r2',r2)
# Результат 73.86% - неплохая модель, лучше, чем с 1 переменной
#вывод графика
X = array[:,12] # x - только степень низкого статуса населения
plt.scatter(X,Y,edgecolors='g')
plt.scatter(X, y_pred,edgecolors='m')
plt.legend(['Реальная','Предсказанная'])
plt.xlabel('Степень низкого статуса населения')
plt.ylabel('Стоимость дома (тыс.долл)')
plt.title('Множественная линейная регрессия от Комарова Д.И.')
plt.show()
#второй график
X = array[:,5] # x - только среднее число комнат
plt.scatter(X,Y,edgecolors='g')
plt.scatter(X, y_pred,edgecolors='m')
plt.legend(['Реальная','Предсказанная'])
plt.xlabel('Среднее число комнам в доме')
plt.ylabel('Стоимость дома (тыс.долл)')
plt.title('Множественная линейная регрессия от Комарова Д.И.')
plt.show()

########################################## Задание 3 ####################################
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
# Загрузка и просмотр данных
# We have to set the column names since it does not come with one
col_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
# Loading the dataset into our dataframe
df = pd.read_csv("auto-mpg.csv", names=col_names, delim_whitespace=True)
# Create features
features = ['weight', 'model_year', 'origin']
X = df[features]
y = df['mpg']
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression()# Initiate mode
lr.fit(X_train, y_train)# Fit model
# Get predictions
test_predict = lr.predict(X_test)# предсказываем пробег автомобиля в милях на галлон на тестовых данных
train_predict = lr.predict(X_train)# предсказываем пробег автомобиля в милях на галлон на обучающих данных
#Оценки качества и точности # Score and compare
y_pred = lr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print('RMSE=',rmse) #среднеквадратическая ошибка
print('коэффициент детерминации(тестовые) r2=',r2_score(y_test, test_predict)) # сравниваем истинные и предсказанные данные
print('коэффициент детерминации(обучающие) r2=',r2_score(y_train, train_predict))  # сравниваем истинные и предсказанные данные
#вывод графика
X = X_test['weight'] # X - только вес автомобиля
Y = y_test
plt.scatter(X,Y,edgecolors='g')
plt.scatter(X, y_pred,edgecolors='m')
plt.legend([ 'Реальная','Предсказанная'])
plt.xlabel('Вес автомобиля')
plt.ylabel('Пробег автомобиля (мили/галлон)')
plt.title('Множественная линейная регрессия  от Комаров Д.И.')
plt.show()
# прогноз для Самойловой. Вот ее автомобиль:
X_test = np.array([[ 2130.0,82.0,2.0]]) #'weight', 'model_year', 'origin'
#Предскажем ей пробег в милях на галлон
test_predict = lr.predict(X_test)# предсказываем пробег автомобиля в милях на галлон
print('Предскажем!!!Пробег автомобиля для Комаров =', test_predict)

########################################### Задание 4 ##########################################
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
# Loading the dataset into our dataframe
# Сведения о 5 параметрах  квартир + цена (столбец 6)
# "wallsMaterial","floorNumber","floorsTotal","totalArea","kitchenArea","latitude","longitude","price"
df = pd.read_csv("moscow_dataset_2020.csv", delimiter=",")
print(df.head())
# Create features
features = ["floorNumber","floorsTotal","totalArea","kitchenArea", "latitude", "longitude",] #Этаж, Число этажей
X = df[features]
y = df['price']
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression() # Initiate mode
lr.fit(X_train, y_train) # Fit model
# Get predictions
test_predict = lr.predict(X_test)# предсказываем цены квартир на тестовых данных
train_predict = lr.predict(X_train)# предсказываем цены квартир на обучающих данных
#Оценки качества и точности # Score and compare
y_pred = lr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print('RMSE=',rmse) #среднеквадратическая ошибка
print('коэффициент детерминации(тестовые) r2=',r2_score(y_test, test_predict)) # сравниваем истинные и предсказанные данные
print('коэффициент детерминации(обучающие) r2=',r2_score(y_train, train_predict))  # сравниваем истинные и предсказанные данные
#вывод графика
X = X_test['totalArea'] # X - только общая площадь
Y = y_test
plt.scatter(X,Y,edgecolors='g')
plt.scatter(X, y_pred,edgecolors='m')
plt.legend([ 'Реальная','Предсказанная'])
plt.xlabel('Площадь квартиры')
plt.ylabel('Цена квартиры (руб * 10 млн.)')
plt.title('Множественная линейная регрессия  от Комарова Д.И.')
plt.show()
 # сохраняем модель в файле на диске
filename = 'finalized_model.sav'
pickle.dump(lr, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
 # прогноз для Самойловой. Вот ее квартира:
X_test = np.array([[2, 5.0, 50.0, 6.0,55.686698,37.595321000000006, ]])
#Предскажем ей цену
test_predict = loaded_model.predict(X_test)# предсказываем цену квартиры
print('Предскажем!!!Цена квартиры для Комарова в рублях =',test_predict)


import pickle
import numpy as np
filename = 'finalized_model.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
 # прогноз для Самойловой. Вот ее квартира:
X_test = np.array([[2, 5.0, 50.0, 6.0,55.686698,37.595321000000006, ]])
#Предскажем ей цену
test_predict = loaded_model.predict(X_test)# предсказываем цену квартиры Самойловой
print('Предскажем!!!Цена квартиры для Комарова в рублях =',test_predict)

###################################### Задание 5 #################################################3
import pandas
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv("Boston.csv", delimiter=",", names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
model = LassoCV()
model.fit(X,Y)
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(Y,y_pred))
r2 = r2_score(Y,y_pred)
print('rmse=', rmse)
print('Точность модели - r2 =',r2)
X = array[:,12] # x - только степень низкого статуса населения
import matplotlib.pyplot as plt
plt.scatter(X,Y,edgecolors='g')
plt.scatter(X, y_pred,edgecolors='m')
plt.legend([ 'Реальная','Предсказанная'])
plt.xlabel('Степень низкого статуса населения')
plt.ylabel('Стоимость дома (тыс.долл)')
plt.title('Регрессия  LASSO от Комарова Д.И.')
plt.show()

###################################### Задание 6 ###################################
import pandas
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv("Boston.csv", delimiter=",", names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
model = ElasticNetCV()
model.fit(X,Y)
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(Y,y_pred))
r2 = r2_score(Y,y_pred)
print('rmse=', rmse)
print('Точность модели - r2 =',r2)
X = array[:,12] # x - только степень низкого статуса населения
import matplotlib.pyplot as plt
plt.scatter(X,Y,edgecolors='g')
plt.scatter(X, y_pred,edgecolors='m')
plt.legend([ 'Реальная','Предсказанная'])
plt.xlabel('Степень низкого статуса населения')
plt.ylabel('Стоимость дома (тыс.долл)')
plt.title('Регрессия  ElasticNet от Комарова Д.И.')
plt.show()

###################################### Задание 7 ###########################################
import pandas
import numpy as np
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv("Boston.csv", delimiter=",", names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
model = DecisionTreeRegressor()
model.fit(X,Y)
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(Y,y_pred))
r2 = r2_score(Y,y_pred)
print('rmse=', rmse)
print('Точность модели - r2 =',r2)
X = array[:,12] # x - только степень низкого статуса населения
import matplotlib.pyplot as plt
plt.scatter(X,Y,edgecolors='g')
plt.scatter(X, y_pred,edgecolors='m')
plt.legend([ 'Реальная','Предсказанная'])
plt.xlabel('Степень низкого статуса населения')
plt.ylabel('Стоимость дома (тыс.долл)')
plt.title('Нелинейная регрессия  DecisionTreeRegressor от Комарова Д.И.')
plt.show()

##################################### Задание 8 ######################################
import pandas
import numpy as np
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv("Boston.csv", delimiter=",", names=names)
X = dataframe[['LSTAT']].values
Y = dataframe['MEDV'].values
model = DecisionTreeRegressor(max_depth=3)
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
plt.title('Нелинейная однофакторная DecisionTreeRegressor от Комарова Д.И.')
plt.show()


import pandas
import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv("Boston.csv", delimiter=",", names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
seed = 7
kfold = model_selection.KFold ( n_splits = 10 , shuffle = True , random_state = 7 )
model = KNeighborsRegressor()
model.fit(X,Y)
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(Y,y_pred))
r2 = r2_score(Y,y_pred)
print('rmse=', rmse)
print('Точность модели - r2',r2)
X = array[:,12] # x - только степень низкого статуса населения
import matplotlib.pyplot as plt
plt.scatter(X,Y,edgecolors='g')
plt.scatter(X, y_pred,edgecolors='m')
plt.legend([ 'Реальная','Предсказанная'])
plt.xlabel('Степень низкого статуса населения')
plt.ylabel('Стоимость дома (тыс.долл)')
plt.title('Нелинейная регрессия KNN  от Комарова Д.И.')
plt.show()

################################# Задание 9 ################################
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

################################## Задание 10 #####################################
import pandas
import numpy as np
from sklearn import model_selection
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv("Boston.csv", delimiter=",", names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
model = SVR(kernel='rbf', C=1e3, gamma=0.1) # SVR(kernel='rbf', C=1e3, gamma=0.1)  SVR(kernel='linear', C=100, gamma="auto")   SVR(kernel='poly', degree=5)
model.fit(X,Y)
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(Y,y_pred))
r2 = r2_score(Y,y_pred)
print('rmse=', rmse)
print('Точность модели - r2',r2)
X = array[:,5] # x - только среднее число комнат
import matplotlib.pyplot as plt
plt.scatter(X,Y,edgecolors='g')
plt.scatter(X, y_pred,edgecolors='m')
plt.legend([ 'Реальная','Предсказанная'])
plt.xlabel('Среднее число комнат в доме')
plt.ylabel('Стоимость дома (тыс.долл)')
plt.title('Нелинейная регрессия SVR от Комарова Д.И.')
plt.show()

######################################## Задание 11 #######################################
import pandas
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv("Boston.csv", delimiter=",", names=names)
array = dataframe.values
X = dataframe[['RM']].values
Y = dataframe['MEDV'].values
model = SVR(kernel='rbf', C=1e3, gamma=0.1) # SVR(kernel='linear', C=1e3)   SVR(kernel='poly', C=1e3, degree=2)
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
plt.legend([ 'Предсказанная','Реальная'])
plt.xlabel('Среднее число комнат в доме')
plt.ylabel('Стоимость дома (тыс.долл)')
plt.title('Нелинейная регрессия SVR от Комарова Д.И.')
plt.show()

##################################### Задание 13 #####################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import operator
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
col_names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
boston_df = pd.read_csv("Boston.csv", names=col_names, delimiter=",")
print(boston_df.head())
array = boston_df.values
X = boston_df[['RM']].values
Y = boston_df['MEDV'].values
#Определение модели============================
#Мы создаем экземпляр LinearRegression и называем его моделью. Затем мы подгоняем наши данные к модели двумя простыми строками:
# Делаем полиномиальное преобразование (X) и совершить предсказание с той же моделью,
pf = PolynomialFeatures(degree=2)
x_poly = pf.fit_transform(X) # Применяем к данным полиномиальные коэффициенты
model = LinearRegression()
model.fit(x_poly, Y) # Обучаем линейную модель полиномиальным  функциям
y_poly_pred = model.predict(x_poly)
rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))
r2 = r2_score(Y,y_poly_pred)
print('rmse=', rmse)
print('Точность модели - r2',r2)
#График
plt.scatter(X, Y, s=10)
plt.scatter(X, y_poly_pred,edgecolors='m')
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.legend([ 'Предсказанная','Реальная'])
plt.xlabel('Среднее число комнат в доме')
plt.ylabel('Стоимость дома (тыс.долл)')
plt.title('Нелинейная полиномиальная регрессия от Комарова Д.И.')
plt.show()