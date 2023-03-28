###################### Задание 1 ########################
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics

X = np.array([
    [13, 161, 2],
    [52, 157, 2],
    [62, 171, 1],
    [0, 168, 0],
    [7, 190, 0],
    [8, 192, 0],
    [12, 165, 1],
    [26, 157, 2],
    [32, 161, 2],
    [10, 174, 0],
    [3, 182, 0],
    [1, 175, 0],
    [51, 150, 2],
    [42, 167, 1],
    [62, 170, 1],
    [4, 178, 0],
    [9, 180, 0],
    [12, 182, 0],
    [35, 155, 1],
    [27, 199, 2],  # Высокая женщина
    [52, 167, 1],
    [40, 184, 0],
    [0, 180, 0],
    [7, 171, 0]])
# должно быть так - 0-тети,  1-дяди
Y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
print('Должно бать (0-тетя, 1-дядя:', Y)
# вывод графиков
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Длинна волос, см')
plt.ylabel('Рост, см')
plt.title('Клиенты от от Комарова: мужчины и женщины')
plt.show()
# 0-женщина  1-Мужчина.
# Interpret 2 cluster solution
model = KMeans(n_clusters=2)
model.fit(X)
y_pred = model.predict(X)
Y_pred = model.labels_  # Y = y
print('Назначила кластеры_________:', y_pred)
# Оценка точности
print("Общая точность кластеризации - accuracy_score =", metrics.accuracy_score(Y_pred, Y))
centroids = model.cluster_centers_
# plot clusters
plt.axis()
# вывод графиков
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')
plt.xlabel('Длинна волос, см')
plt.ylabel('Рост, см')
plt.title('Два кластера от Комарова: мужчины и женщины')
plt.show()
# ROC-анализ
from sklearn.metrics import roc_curve, auc

actual = Y
predictions = y_pred
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic from Komarov')
plt.plot(false_positive_rate, true_positive_rate, 'b',
         label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

######################### Задание 2 #######################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split  # для кросс-валидации

print("КЛАСТЕРИЗАЦИЯ ОТ Комарова Д.И. - ДВА кластера")
# Сведения о выплате клиентами кредита: 0- Не выплатил (Высокий риск)  1- Выплатил (Низкий риск выдачи кредита)
data = pd.read_csv('blackjack.csv',  # исходный файл - обучающая выборка + предполагаемые классы
                   sep=',', header=None)
Y = np.array(data[3])  # последний столбец - кластеры, так должно быть, в алгоритме не участвует, только для точности
del data[3]  # Убрали из основной выборки. Там остались только признаки:возраст, доход, недвижимость
X = np.array(data)
# Реализуем кросс - валидацию. К=0.4
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
# вывод графиков
plt.scatter(x_test[:, 0], x_test[:, 1])
plt.xlabel('возраст')
plt.ylabel('доход')
plt.title('Клиенты от Комарова')
plt.show()
# Строим модель кластеризации
model = KMeans(n_clusters=2)
# Обучаем модель только на свойствах (без классов)
model.fit(x_train)
# предсказываем классы
result = model.predict(x_test)
# Compare predicted class labels with actual class labels
n_predicted_correctly = np.sum(result == y_test)
print("напрогнозировала:", result)
print("а должно бы быть:", y_test)
print("размер    выборки:", y_test.size)
print("из них корректных:", n_predicted_correctly)
print("Кросс-валидация. Результаты оценки качества модели кластеризации:")
print("Общая точность кластеризации - accuracy_score =", metrics.accuracy_score(y_test, result))
print("По классам:  Точность -  Полнота - F-мера -  Кол-во")
print(metrics.classification_report(y_test, result))
centroids = model.cluster_centers_
# plot clusters
plt.axis()
plt.scatter(x_test[:, 0], x_test[:, 1], c=result)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')
plt.xlabel('Первая карта')
plt.ylabel('Вторая карта')
plt.title('Кластеризация от Комарова: Риск проигрыша')
plt.show()
# ROC-анализ
from sklearn.metrics import roc_curve, auc

actual = y_test
predictions = result
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic from Komarov')
plt.plot(false_positive_rate, true_positive_rate, 'b',
         label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

################################## Задание 3 ########################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split  # для кросс-валидации

print("КЛАСТЕРИЗАЦИЯ ОТ Комарова Д.И. - ДВА КЛАССА")
# Загружаем набор данных Ирисы:
# Сведения о параметрах  ирисов 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
# Сведения о классах, к которым должны относиться  ирисы (целевые переменные) 0- 'setosa', 1-'versicolor'
data = pd.read_csv('iris_claster_bin.csv',  # исходный файл - обучающая выборка + предполагаемые кластеры
                   sep=',', header=None)
Y = np.array(data[4])  # последний столбец - классы для обучения
del data[4]  # Убрали из основной выборки. Там остались только признаки
X = np.array(data)
# Реализуем кросс - валидацию. К=0.4
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# вывод графика 1
plt.axis()
plt.scatter(x_test[:, 0], x_test[:, 1])
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('ИРИСЫL  от Комарова Д.И.')
plt.show()
# Строим бинарную модель кластеризации
model = KMeans(n_clusters=2)
# Обучаем модель только на свойствах (без классов)
model.fit(x_train)
# предсказываем классы
result = model.predict(x_test)
# Compare predicted class labels with actual class labels
n_predicted_correctly = np.sum(result == y_test)
print("напрогнозировала:", result)
print("а должно бы быть:", y_test)
print("размер    выборки:", y_test.size)
print("из них корректных:", n_predicted_correctly)
print("Кросс-валидация. Результаты оценки качества модели кластеризации:")
print("Общая точность кластеризации - accuracy_score =", metrics.accuracy_score(y_test, result))
print("По классам:  Точность -  Полнота - F-мера -  Кол-во")
print(metrics.classification_report(y_test, result))
# вывод графика 1
centroids = model.cluster_centers_
plt.axis()
plt.scatter(x_test[:, 0], x_test[:, 1], c=result)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('ИРИСЫL. Кластеризация K-Meanse от Комарова Д.И.')
plt.show()
# вывод графика 2
plt.scatter(x_test[:, 2], x_test[:, 3], c=result)
plt.scatter(centroids[:, 2], centroids[:, 3], c='red', s=100, marker='x')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('ИРИСЫL. Кластеризация K-Meanse от Комарова Д.И.')
plt.show()
# ROC-анализ
from sklearn.metrics import roc_curve, auc

actual = y_test
predictions = result
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic from Komarov')
plt.plot(false_positive_rate, true_positive_rate, 'b',
         label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

####################### Задание 4 #############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split  # для кросс-валидации

print("КЛАСТЕРИЗАЦИЯ ОТ Комарова Д.И. - ТРИ КЛАССА")
# Загружаем набор данных Ирисы:
# Сведения о параметрах  ирисов 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
# Сведения о кластерах, к которым должны относиться  ирисы  0- 'setosa', 1-'versicolor', 2-'virginica'
data = pd.read_csv('iris_claster.csv',  # исходный файл - обучающая выборка + предполагаемые классы
                   sep=',', header=None)
Y = np.array(data[4])  # последний столбец - кластеры должны стать такими
del data[4]  # Убрали из основной выборки. Там остались только признаки
X = np.array(data)
# Реализуем кросс - валидацию. К=0.4
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
# вывод графика (кластизации пока нет)
xs = x_test[:, 0]
ys = x_test[:, 1]
plt.scatter(xs, ys)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('ИРИСЫ (тестовая выборка) от Комарова Д.И.')
plt.show()
# Строим модель кластеризации
model = KMeans(n_clusters=3)
# Обучаем модель только на свойствах
model.fit(x_train)
# предсказываем кластеры
result = model.predict(x_test)
# Compare predicted labels with actual  labels
n_predicted_correctly = np.sum(result == y_test)
print("напрогнозировала:", result)
print("а должно бы быть:", y_test)
print("размер    выборки:", y_test.size)
print("из них корректных:", n_predicted_correctly)
print("Кросс-валидация. Результаты оценки качества модели кластеризации:")
print("Общая точность кластеризации - accuracy_score =", metrics.accuracy_score(y_test, result))
print("По кластерам:  Точность -  Полнота - F-мера -  Кол-во")
print(metrics.classification_report(y_test, result))
# вывод графика 1
centroids = model.cluster_centers_
plt.axis()
plt.scatter(x_test[:, 0], x_test[:, 1], c=result)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('ИРИСЫL. Кластеризация K-Meanse от Комарова Д.И.')
plt.show()
# вывод графика 2
plt.scatter(x_test[:, 2], x_test[:, 3], c=result)
plt.scatter(centroids[:, 2], centroids[:, 3], c='red', s=100, marker='x')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('ИРИСЫL. Кластеризация K-Meanse от Комарова Д.И.')
plt.show()

############################### Задание 5 #######################
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# вывод графика  - - зависимость расхода от дохода
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Годовой доход, тыс.$')
plt.ylabel('Годовой расход, тыс.$')
plt.title('От Комарова: клиенты магазина')
plt.show()
model = KMeans(n_clusters=5)
model.fit(X)
y_pred = model.predict(X)
print('Предсказала кластеры')
print(y_pred)
centroids = model.cluster_centers_
# plot clusters
plt.axis()
# вывод графика кластеров
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')
plt.xlabel('Годовой доход, тыс.$')
plt.ylabel('Годовой расход, тыс.$')
plt.title('Кластеризация от Комарова: клиенты магазина')
plt.show()
# Выбор оптимального числа кластеров
inertia = []  # Для каждого значения кластеров подсчитаем инерцию
# Инерция -  Сумма квадратов расстояний до ближайшего центра кластера
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
    inertia.append(np.sqrt(kmeans.inertia_))
plt.plot(range(1, 8), inertia, marker='s');
plt.xlabel('$k$ - число кластеров')
plt.ylabel('Inertia- сумма квадратов расстояний до ближайшего центра кластера');
plt.show()

################################ Задание 6 ###############################
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# Создать образец данных
# вывод графика  - - зависимость расхода от дохода
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Годовой доход, тыс.$')
plt.ylabel('Годовой расход, тыс.$')
plt.title('От Комарова: клиенты магазина')
plt.show()
model = AffinityPropagation(random_state=1)
model.fit(X)
y_pred = model.predict(X)
print('Предсказала кластеры')
print(y_pred)
centroids = model.cluster_centers_
# plot clusters
plt.axis()
# вывод графика кластеров
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')
plt.xlabel('Годовой доход, тыс.$')
plt.ylabel('Годовой расход, тыс.$')
plt.title('Кластеризация от Комарова: клиенты магазина')
plt.show()
cluster_centers_indices = model.cluster_centers_indices_
labels = model.labels_
n_clusters_ = len(cluster_centers_indices)
print('Число кластеров =', n_clusters_)
# Средний коэффициент силуета - точность д.б. от 0 до 1
print("Средний коэффициент силуета = ", metrics.silhouette_score(X, labels, metric='sqeuclidean'))

################################# Задание 7 ########################
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

dataset = pd.read_csv('cust_seg.csv')
dataset.drop('Num', axis=1, inplace=True)  # Удаляем столбец с номерами
print(dataset.head())
# DATA PROCESSING
X = dataset.values
X = np.nan_to_num(X)  # Функция nan_to_num() заменяет nan на 0
X = dataset.iloc[:, [1, 2, 4, 8]].values  # отбираем только возраст, уровень образования, доход, дополн_доход
print(X)
# вывод графика  - - зависимость расхода от дохода
plt.scatter(X[:, 0], X[:, 2])  # возраст-доход
plt.xlabel('Возраст')
plt.ylabel('Годовой доход, тыс.$')
plt.title('От Комарова: клиенты банка')
plt.show()
model = KMeans(n_clusters=5)
model.fit(X)
y_pred = model.predict(X)
print('Предсказала кластеры')
print(y_pred)
centroids = model.cluster_centers_
# plot clusters
plt.axis()
# вывод графика кластеров
plt.scatter(X[:, 0], X[:, 2], c=model.labels_)  # возраст-доход
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')
plt.xlabel('Возраст')
plt.ylabel('Годовой доход, тыс.$')
plt.title('Кластеризация от Комарова: клиенты банка')
plt.show()
# Выбор оптимального числа кластеров
inertia = []  # Для каждого значения кластеров подсчитаем инерцию
# Инерция -  Сумма квадратов расстояний до ближайшего центра кластера
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
    inertia.append(np.sqrt(kmeans.inertia_))
plt.plot(range(1, 8), inertia, marker='s');
plt.xlabel('$k$ - число кластеров')
plt.ylabel('Inertia- сумма квадратов расстояний до ближайшего центра кластера');
plt.show()

##############################  Задание 8 ########################
import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the image
image = cv2.imread("me.jpg")  # my_test.jpg
# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)
# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# number of clusters (K)
k = 2  # чИСЛО кластеров
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# convert back to 8 bit values
centers = np.uint8(centers)
# flatten the labels array
labels = labels.flatten()
# reshape back to the original image dimension
# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)
# show the image
plt.imshow(segmented_image)
plt.show()

########################## Задание 9 #############################
import PIL.Image as image
from skfuzzy.cluster import cmeans
from pylab import *


def loadData(filepath):
    f = open(filepath, 'rb')
    data = []
    img = image.open(f)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel((i, j))
            data.append([x / 256.0, y / 256.0, z / 256.0])
    f.close()
    return np.mat(data), m, n


imgData, row, col = loadData('me.jpg')  # my_test.jpg
print(imgData, row, col)
imgData = imgData.T
# c - количество кластеров, которые нужно указать.
# m является указанным выше индексом членства, который является взвешенным индексом.
center, u, u0, d, jm, p, fpc = cmeans(imgData, m=2, c=3, error=0.0001, maxiter=1000)
for i in u:
    label = np.argmax(u, axis=0)  # Получить максимальное значение столбца
label = label.reshape([row, col])  # преобразованное измерение
print(fpc)
print(center)
print(np.max(label))
pic_new = image.new('L', (row, col))
print(pic_new)
# Наконец, используйте значение RGB центральной точки кластера, чтобы заменить значение каждого пикселя
# в исходном изображении и получить окончательное сегментированное изображение.
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
pic_new.save('result.jpg')
plt.imshow(pic_new)
plt.show()

####################### Задание 10 ########################3
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd

# Загружаем набор данных Ирисы:
# Сведения о кластерах, к которым должны относиться  ирисы  0- 'setosa', 1-'versicolor', 2-'virginica'
df = pd.read_csv('iris_claster.csv', sep=',', header=None)  # исходный файл - обучающая выборка + классы
df.drop(df.columns[[4]], axis=1)  # удаляем последний столбец классов
# plotting dendrogram # Linkage Matrix
plt.figure(figsize=(10, 7))
dendrogram(linkage(df, method='ward'))  # ,color_threshold = 3
plt.title('Дендограмма Комарова ')
plt.ylabel('Euclidean distance')
plt.xlabel('Ирисы (параметры)')
plt.show()

################################### Задание 11 #############################
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd

df = pd.read_csv('blackjack.csv', sep=',')  # Кредиты банка
df.drop(df.columns[[3]], axis=1)  # удаляем последний столбец классов
# plotting dendrogram # Linkage Matrix
plt.figure(figsize=(12, 7))
dendrogram(linkage(df, method='ward'))  # ,color_threshold = 3
plt.title('Дендограмма Комарова ')
plt.ylabel('Euclidean distance')
plt.xlabel('клиенты банка (параметры)')
plt.show()

################################## Задание 12 #############################3
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

img = cv2.imread('me.jpg')  # my_test.jpg
Z = np.float32(img.reshape((-1, 3)))
db = DBSCAN(eps=0.3, min_samples=100).fit(Z[:, :2])
plt.imshow(np.uint8(db.labels_.reshape(img.shape[:2])))
plt.show()
