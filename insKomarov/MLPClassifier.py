from sklearn.neural_network import MLPClassifier
import numpy as np
x_train = [[0, 0],
      [0, 1],
      [1, 0],
      [1, 1]]
y_train = [0, 1, 1, 0]    #.reshape(4,)
model = MLPClassifier(activation='relu', max_iter=10000, hidden_layer_sizes=(4,2)) #'logistic'
model.fit(x_train, y_train)
x_test = x_train
y_test = y_train
print('score:', model.score(x_test, y_test)) # outputs 0.5 - logistic'
print('predictions:', model.predict(x_test)) # outputs [0, 0, 0, 0] - logistic'
print('expected:', np.array([0, 1, 1, 0]))

####################### Задание 3 #######################
from sklearn.neural_network import MLPClassifier
x_train = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_train = [0, 0, 0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
print(clf.fit(x_train, y_train))
#Атрибут coefs_ содержит список весовых матриц для каждого слоя.
print("Весовые матрицы синапсов")
print("веса между входным и первым скрытым слоем : weights between input and first hidden layer:")
print(clf.coefs_[0])
print("\веса между первым скрытым и вторым скрытым слоем:nweights between first hidden and second hidden layer:")
print(clf.coefs_[1])
#ТЕСТОВЫЕ ДАННЫЕ
x_test = [[0, 0],[0, 1],[1, 0],[0, 1],[1, 1],[2., 2.],[1.3, 1.3], [2, 4.8]]
result = clf.predict(x_test)
print('Найденные классы:',result)

######################## Задание 4 #########################
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
print("КЛАССИФИКАЦИЯ - ТРИ КЛАССА")
# Загружаем набор данных Ирисы:
# Сведения о параметрах  ирисов 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
# Сведения о классах, к которым относят  ирисы (целевые переменные) 0- 'setosa', 1-'versicolor', 2-'virginica'
data = pd.read_csv('iris_all.csv',  # исходный файл - обучающая выборка + классы
                          sep=',', header=None)
y_train = np.array(data[4]) # последний столбец - классы для обучения
del data[4]   # Убрали из основной выборки. Там остались только признаки
x_train = np.array(data)
# Смотрим на данные, выводим 10 первых строк:
print(x_train[:10])
# Смотрим на целевые переменные:
print(y_train)
x_test = x_train  # тестовая выборка равна обучающей
y_test = y_train # планируемые результаты - классы из обучающей выбоки
#создаем экземпляр объекта классификационной модели
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,4,15),max_iter=1000) #Взять 100,200,300,400
#обучаем модель на конкретных обучающих данных
model.fit(x_train, y_train)
#получаем результат для тестовых данных
result = model.predict(x_test)
print('Результат прогноза (class):',result)
# число правильных тестов из всей тестовой выборки
n_predicted_correctly = np.sum(result == y_test)
print("размер    выборки:",y_test.size)
print("из них корректных:",n_predicted_correctly)
print("Результаты оценки качества модели прогнозирования:")
print("точность классификации - accuracy_score =", metrics.accuracy_score(y_test, result))
print("             Точность - Полнота -  F-мера - кол-во")
print(metrics.classification_report(y_test, result))

######################### Задание 5 ########################
from sklearn.datasets import load_digits#Каждая точка данных представляет собой изображение 8x8 цифры.
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import cv2 as cv # pip install opencv-python
# load data
digits = load_digits()
print('We have %d samples'%len(digits.target))
## Вывод первых 32 изображений - введенные данные
fig = plt.figure(figsize = (8,8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(32):
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))
#++++++++++++++++++++++++
# split data to training and testing data
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=16)
print('Number of samples in training set: %d, number of samples in test set: %d'%(len(y_train), len(y_test)))
# Initialize ANN classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,30,20), activation='logistic', max_iter = 1000) #можно 30,30,30
# Train the classifier with the traning data
mlp.fit(X_train,y_train)
# predict results from the test data - это изображение №13  от начала выборки
X_test_1 = np.array([digits.data[13]]) # Это 3
plt.matshow(digits.images[13])
plt.show()
predicted = mlp.predict(X_test_1)
print('Распознала!! Это - ',predicted)
# predict results from the test data - это изображение №13  от начала выборки
X_test_2 = np.array([digits.data[31]]) # Это 9
plt.matshow(digits.images[31])
plt.show()
predicted = mlp.predict(X_test_2)
print ('Распознала!! Это - ',predicted)
img = cv.imread('pic1.png') # размер изо = 8x8 пикс
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY,0)
img = img.astype('float32')
plt.matshow(img)
plt.show()
pred_img = img.flatten().reshape(1, 64)
X_test = np.array(pred_img) # Э
print('Распознала рисунок !! Это - ', mlp.predict(X_test))

