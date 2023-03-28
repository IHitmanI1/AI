##################### Задание 1 #######################
import numpy as np
from sklearn.linear_model import Perceptron
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
# Загружаем набор данных Ирисы:
# Сведения о параметрах  ирисов 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
# Сведения о классах, к которым относят  ирисы (целевые переменные) 0- 'setosa', 1-'versicolor'
print("БИНАРНАЯ КЛАССИФИКАЦИЯ")
data = pd.read_csv('iris_all_binary.csv',  # исходный файл - обучающая выборка + классы
                          sep=',', header=None)
y_train = np.array(data[4]) # последний столбец - классы для обучения
del data[4]   # Убрали из основной выборки. Там остались только признаки
x_train = np.array(data)
# Смотрим на данные, выводим 10 первых строк:
print (x_train[:10])
# Смотрим на целевые переменные:
print (y_train)
y_test = y_train
x_test = x_train
model = Perceptron(random_state=42, max_iter=10)
model.fit(x_train, y_train)
print(model)
result = []
for value in x_test:
    pred = model.predict([value])
    result.append(pred[0])
result = np.array(result)
print(result)
n_predicted_correctly = np.sum(result == y_test)
print("напрогнозировала:",result)
print("а должно бы быть:",y_test)
print("размер выборки:",y_test.size)
print("из них корректных:",n_predicted_correctly)
print ("Результаты оценки качества модели прогнозирования:")
print ("Общая точность классификации - accuracy_score =", metrics.accuracy_score(y_test, result))
print ("По классам:  Точность -  Полнота - F-мера -  Кол-во")
print (metrics.classification_report(y_test, result))
#вывод графиков
name = ['setosa', 'versicolor']
colors = ['red', 'blue']
for i in [0, 1]:
   xs = x_test[:, 0][result == i]
   ys = x_test[:, 1][result == i]
   plt.scatter(xs, ys, c=colors[i])
   plt.legend(name)
   plt.xlabel('sepal length (cm)')
   plt.ylabel('sepal width (cm)')
plt.title('Класс ИРИСОВ от SEPAL. ANN Perceptron')
plt.show()
for i in [0, 1]:
   xs = x_test[:, 2][result == i]
   ys = x_test[:, 3][result == i]
   plt.scatter(xs, ys, c=colors[i])
   plt.legend(name)
   plt.xlabel('petal length (cm)')
   plt.ylabel('petal width (cm)')
plt.title('Класс ИРИСОВ от PETAL. ANN Perceptron')
plt.show()

#################### Задание 2 ###############################
import numpy as np
from sklearn.linear_model import Perceptron
#Алгоритм ANN Perceptron бинарная классификация
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
print("БИНАРНАЯ КЛАССИФИКАЦИЯ")
# Сведения о выплате клиентами кредита: class=0- Не выплатил (Высокий риск)  =1- Выплатил (Низкий риск,клиент хороший)
#names = ['age','salary','house','class'] - возраст, зарплата, наличие недвижимости
dataframe = pd.read_csv('blackjack.csv',  # исходный файл - обучающая выборка + классы
                          sep=',',header=0)
print("Фрагмент обучающих данных")
print (dataframe.head(10))
array = dataframe.values
x_train = array[:,0:3] #все строки, столбцы 0,1,2 - до третьего
y_train = array[:,3]  # все строки четвертого столбца
# Смотрим на целевые переменные:
print (y_train)
y_test = y_train
x_test = x_train
model = Perceptron(random_state=42, max_iter=10) #Perceptron(tol=1e-3)
model.fit(x_train, y_train)
print(model)
result = []
for value in x_test:
    pred = model.predict([value])
    result.append(pred[0])
result = np.array(result)
print(result)
n_predicted_correctly = np.sum(result == y_test)
print("напрогнозировала:",result)
print("а должно бы быть:",y_test)
print("размер выборки:",y_test.size)
print("из них корректных:",n_predicted_correctly)
print ("Результаты оценки качества модели прогнозирования:")
print ("Общая точность классификации - accuracy_score =", metrics.accuracy_score(y_test, result))
print ("По классам:  Точность -  Полнота - F-мера -  Кол-во")
print (metrics.classification_report(y_test, result))
#вывод графиков
name = ['Высокий риск', 'Низкий риск']
colors = ['red', 'blue']
for i in [0, 1]:
   xs = x_test[:, 0][result == i]
   ys = x_test[:, 1][result == i]
   plt.scatter(xs, ys, c=colors[i])
   plt.legend(name)
   plt.xlabel('Первая карта')
   plt.ylabel('Вторая карта')
plt.title('Риски проигрыша. ANN Perceptron')
plt.show()

