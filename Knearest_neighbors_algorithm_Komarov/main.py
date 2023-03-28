import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import metrics

# Сведения о выплате клиентами кредита: class=0- Не выплатил (Высокий риск, плохой клиент)  =1- Выплатил (Низкий риск,клиент хороший)
# names = ['age','salary','house','class'] - возраст, зарплата, наличие недвижимости
dataframe = pd.read_csv('blackjack.csv',  # исходный файл - обучающая выборка + классы
                        sep=',')
print('Введены поля для обработки:', dataframe.columns[0], dataframe.columns[1], dataframe.columns[2],
      dataframe.columns[3])
# вывод 10 первых строк
print("Фрагмент обучающих данных")
print(dataframe.head(10))
array = dataframe.values
x_train = array[:, 0:3]  # все строки, столбцы 0,1,2 - до третьего
y_train = array[:, 3]  # все строки четвертого столбца
x_test = np.array([
    [22, 14, 0],
    [24, 26, 1],
    [27, 45, 1],
    [20, 12, 0],
    [30, 55, 1]
])
x_test = x_train  # тестовая выборка равна обучающей
y_test = y_train  # планируемые результаты - классы из обучающей выбоки
# создаем экземпляр объекта классификационной модели
model = KNeighborsClassifier(n_neighbors=3)  # , algorithm = 'auto', weights = "distance",  metric = "euclidean")
# обучаем модель на конкретных обучающих данных
model.fit(x_train, y_train)
# получаем результат для тестовых данных
result = model.predict(x_test)
print("Тестовые данные")
print('card1', 'card2', 'blackjack', 'win')
print(x_test)
print('Результат прогноза (class):', result)

# число правильных тестов из всей тестовой выборки
n_predicted_correctly = np.sum(result == y_test)
print("размер    выборки:", y_test.size)
print("из них корректных:", n_predicted_correctly)
print("Результаты оценки качества модели прогнозирования:")
print("точность классификации - accuracy_score =", metrics.accuracy_score(y_test, result))
print("             Точность - Полнота -  F-мера - кол-во")
print(metrics.classification_report(y_test, result))

name = ['Высокий риск', 'Низкий риск']
colors = ['red', 'blue']
for i in [0, 1]:
    xs = x_test[:, 0][result == i]
    ys = x_test[:, 1][result == i]
    plt.scatter(xs, ys, c=colors[i])
    plt.legend(name)
    plt.xlabel('Первая карта')
    plt.ylabel('Вторая карта')
plt.title('Риски проигрыша. Метод k-СОСЕДЕЙ')
plt.show()
