import pandas as pd
from sklearn.model_selection import train_test_split # pip install scikit-learn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from neupy.algorithms import PNN
# Загрузка и просмотр данных
#Поля:RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,
# Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
dataframe = pd.read_csv("Churn_Modelling.csv")
dataframe.head()
#Преобразование данных
dataframe['Geography'].replace("France",1,inplace= True)
dataframe['Geography'].replace("Spain",2,inplace = True)
dataframe['Geography'].replace("Germany",3,inplace=True)
dataframe['Gender'].replace("Female",0,inplace = True)
dataframe['Gender'].replace("Male",1,inplace=True)
print(dataframe)
# Создание корреляционной матрицы
#Корреляционная матрица показывает, какие параметры будут влиять на результат.
# Сразу можно выделить 3 положительные корреляции: «Баланс счета», «Возраст»,
# «Географическое положение». Balance, Age, Geography
correlation = dataframe.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between different fearures')
plt.show()
#Кросс валидация
array = dataframe.values
X = array[:, 3:13] #все строки, столбцы 4,5,6 - включая 13=ый
Y = array[:, 13]  # все значения 14-го столбца
Y = Y.astype('int')  #чтобы не было ошибки в данных
# Для избежания проблем с переобучением разделим наш набор данных:
# когда вы используете 42, вы всегда получите один и тот же результат при первом выполнении разделения.
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.4, random_state=42) # 42,0,21
#Прогноз
model = PNN(std=12) #  стандартное отклонение плотности вероятности  порядка входных данных
model.train(X_train, y_train)
result = model.predict(X_test)
from sklearn import metrics
print("Классификация.Вероятностная ИНС от Самойловой")
print("напрогнозировала:",result)
print("а должно бы быть:",y_test)
print("размер    выборки:",y_test.size)
print("Результаты оценки качества модели прогнозирования:")
print("             Точность - Полнота -  F-мера - кол-во")
print(metrics.classification_report(y_test, result))

x_test = np.array([[376,3,0,49,4,115046.74,4,1,0,119346.88]])  # это Самойлова, она в этом банке
predicted = model.predict(x_test)
print('Прогноз ухода из банка =', predicted)

######################## Задание 2 #################################
import numpy as np
import pandas as pd
from sklearn import metrics
from neupy import algorithms  # PNN
from sklearn.model_selection import train_test_split # pip install scikit-learn
print("КЛАССИФИКАЦИЯ - ТРИ КЛАССА")
# Загружаем набор данных Ирисы:
# Сведения о четырех параметрах  ирисов 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
# Сведения о классах, к которым относят  ирисы (целевые переменные) 0- 'setosa', 1-'versicolor', 2-'virginica'
data = pd.read_csv('iris_all.csv',  # исходный файл - обучающая выборка + классы
                          sep=',', header=None)
array = data.values
X = array[:, 0:4] #все строки, столбцы 0-3
Y = array[:, 4]  # все значения 5-го столбца
Y = Y.astype('int')  #чтобы не было ошибки в данных
# Для избежания проблем с переобучением разделим наш набор данных:
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.4, random_state=0) # 42,0,21
#задаем модель ИНС
print ("Обучается вероятностная нейронная сеть - PNN")
print ("Использует радиальные базисные функции как функции активации")
model = algorithms.PNN(verbose=False, std=5)   #  стандартное отклонение плотности вероятности порядка входных данных
model.train(x_train, y_train)
model_predictions = model.predict(x_test)
print ("Результаты оценки качества прогнозирования:")
print ("accuracy_score \оценка точности\ =", metrics.accuracy_score(y_test, model_predictions))
print('точность - полнота - F-мера - число экземпляров в тестовой выборке')
print (metrics.classification_report(y_test, model_predictions))
# Предсказание класса на тестовой выборке (три ириса):
test_data1 = np.array([
       [ 5.1,  3.5,  1.4,  0.2],  # первый ирис - 0 класс
       [6.9,   3.1,   5.4, 2.1],  # второй ирис - 2 класс
       [ 5.9,  3. ,  5.1,  1.8]])  # третий ирис - 1 класс
print("Тестовые данные:")
print(test_data1)
print("Прогноз для тестовых данных (по двум ирисам):",model.predict(test_data1))
print("Hello, Student SMOLGU!!!!")

##################### Задание 3 ########################
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from neupy import algorithms
# Загрузка и просмотр данных
# We have to set the column names since it does not come with one
col_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
# Loading the dataset into our dataframe
df = pd.read_csv("auto-mpg.csv", names=col_names, delim_whitespace=True)
# Create features
features = ['weight', 'model_year', 'origin']
X1 = df[features]
y1 = df['mpg']
X = np.array(X1)
y = np.array(y1)
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
nn = algorithms.GRNN(std = 20, verbose = True) # verbose - многофакторная
#Сеть использует ленивое обучение, которое означает, что сети не требуется итеративное обучение.
nn.train(X_train, y_train)
y_pred = nn.predict(X_test)
#средний квадрат ошибки  RMSE
mse = np.mean((y_pred - y_test) ** 2)
print ("Среднеквадратичная ошибка:", mse) # оценка погрешности Mean Squared Error (см.мою лекцию)
print ("Предсказываю пробег для тестовой выборки:")
print (y_pred)
print("OK!!")
#ГРАФИК
name = ['Обучение', 'Прогноз']
plt.scatter(X_train[:,0], y_train, color='black')
plt.scatter(X_test[:,0], y_pred,  color='red');
plt.xlabel('weight')
plt.ylabel('Пробег автомобиля в милях на галлон')
plt.title('Прогноз пробега. Метод GRNN - ИНС')
plt.legend(name)
plt.show()
X_test = np.array([[ 2130.0,82.0,2.0]]) #'weight', 'model_year', 'origin'
#Предскажем ей пробег в милях на галлон
test_predict = nn.predict(X_test)# предсказываем пробег автомобиля в милях на галлон
print('Предскажем!!!Пробег автомобиля =',test_predict)



