import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

# Сведения о выплате клиентами кредита: class=0- Не выплатил (Высокий риск)  =1- Выплатил (Низкий риск,клиент хороший)
#names = ['age','salary','house','class'] - возраст, зарплата, наличие недвижимости
dataframe = pd.read_csv('DATA/blackjack.csv',  # исходный файл - обучающая выборка + классы
                          sep=',')
#вывод 10 первых строк
print("Фрагмент обучающих данных")
print(dataframe.head(10))
array = dataframe.values
x_train = array[:,0:3] #все строки, столбцы 0,1,2 - до третьего
y_train = array[:,3]  # все строки четвертого столбца
x_test = x_train  # тестовая выборка равна обучающей
y_test = y_train # планируемые результаты - классы из обучающей выбоки
#создаем экземпляр объекта классификационной модели
model = GaussianNB()
#обучаем модель на конкретных обучающих данных
model.fit(x_train, y_train)
#получаем результат для тестовых данных
result = model.predict(x_test)
print('Результат прогноза (class):',result)
# число правильных тестов из всей тестовой выборки
n_predicted_correctly = np.sum(result == y_test)
print("размер    выборки:",y_test.size)
print("из них корректных:",n_predicted_correctly)
print ("Результаты оценки качества модели прогнозирования:")
print ("точность классификации - accuracy_score =", metrics.accuracy_score(y_test, result))
print ("             Точность - Полнота -  F-мера - кол-во")
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
plt.title('Риски проигрыша. Метод  Наивного Байеса от Комарова')
plt.show()
# ROC-анализ
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
probs = model.predict_proba(x_test) #степень уверенности в ответе, вероятность 0 и 1 (две колонки)
#print(probs)
# keep probabilities for the positive outcome only
probs = probs[:, 1] # отбираем вероятности для 1
# calculate AUC
my_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % my_auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.title('Receiver Operating Characteristic from Komarov')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# show the plot
plt.show()

############################################# Задание 2 ################################################################
Weather = np.array([[0], [1], [2], [0], [0], [1], [2], [2], [0], [2], [0], [1], [1], [2]])
Play = np.array([0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0])
clf = GaussianNB()
clf.fit(Weather, Play)
print('Прогноз удачной игры' ,clf.predict([[2]])) #Задаем погоду
print('Точность:',clf.score(Weather, Play))

############################################## Задание 3 ###############################################################
df = pd.read_csv('DATA/diabetes.csv')  # исходный файл - обучающая выборка + классы
print("Фрагмент обучающих данных")
print(df.head(10))
feature_cols = ['Pregnancies','Glucose','Insulin','BMI','Age']
X = df[feature_cols] # Features
y = df.Outcome # Target variable
#код разделит набор данных на 70% данных обучения и 30% данных тестирования.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
#обучаем модель с помощью класса sklearn класса DecisionTreeClassifier
clf = GaussianNB()
clf = clf.fit(X_train,y_train)
#нужно сделать прогноз
y_pred = clf.predict(X_test)
# получим оценку точности, матрицу путаницы и отчет о классификации следующим образом:
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
plot_confusion_matrix(clf, X_test, y_test)
plt.show()
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
# ROC-анализ
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
probs = clf.predict_proba(X_test) #степень уверенности в ответе, вероятность 0 и 1 (две колонки)
#print(probs)
# keep probabilities for the positive outcome only
probs = probs[:, 1] # отбираем вероятности для 1
# calculate AUC
my_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % my_auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.title('Receiver Operating Characteristic from Komarov')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# show the plot
plt.show()

#######################################################################################################################
from sklearn.model_selection import train_test_split
print("КЛАССИФИКАЦИЯ ОТ Комарова - Семь КЛАССОВ")
# 4.   Информация об атрибутах:
# 1. Идентификационный номер: от 1 до 214
# 2. RI: показатель преломления света
# 3. Na: натрий (единица измерения: массовые проценты в соответствующем оксиде, как и атрибуты 4-10)
# 4. Mg: магний
# 5. Al: алюминий
# 6. Si: Кремний
# 7. K: Калий
# 8. Ca: Кальций
# 9. Ba: Барий
# 10. Fe: Железо
# 11. Тип стекла: (класс)
#1.    оконное стекло (термополированное)
# 2. оконное стекло (нетермополированное)
# 3. автомобильное стекло (термополированное)
# 4. автомобильное стекло (нетермополированное) – этот тип отсутствует в данных
# 5. стекло для контейнеров
# 6. посудное
# 7. фары
data = pd.read_csv('DATA/glass.csv', sep=',', header=None)
Y = np.array(data[10]) # последний столбец - классы для обучения
del data[10]   # Убрали из основной выборки. Там остались только признаки
X = np.array(data)
import seaborn as sns  #Корреляционная матрица
correlation = data.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between different fearures')
plt.show()
# Реализуем кросс - валидацию. К=0.33
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
clf = GaussianNB()
clf.fit(x_train, y_train)
# Make class predictions for all observations in X
y_pred = clf.predict(x_test)
# Compare predicted class labels with actual class labels
n_predicted_correctly = np.sum(y_pred == y_test)
print("напрогнозировала:",y_pred)
print("а должно бы быть:",y_test)
print("размер    выборки:",y_test.size)
print("из них корректных:",n_predicted_correctly)
print ("Результаты оценки качества модели прогнозирования:")
print ("Общая точность классификации - accuracy_score =", metrics.accuracy_score(y_test, y_pred))
print ("По классам:  Точность -  Полнота - F-мера -  Кол-во")
print (metrics.classification_report(y_test, y_pred))
# получим оценку точности, матрицу путаницы и отчет о классификации следующим образом:
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, x_test, y_test)
plt.title('Confusion Matrix')
plt.show()
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:", result2)

################################################### Регрессия ##########################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#цель - построить модель логистической регрессии на Python, чтобы определить, будут ли кандидаты приняты в престижный университет.
#результат GMAT, средний балл успеваемости и количество лет опыта работы. Класс - будет ли принят в университет
candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }
df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])
#print (df)
X = df[['gmat', 'gpa','work_experience']]
y = df['admitted']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
logistic_regression= LogisticRegression(solver='liblinear', random_state=0) # алгоритм оптимизации, решатель (solver)='liblinear' 'newton-cg ' или 'lbfgs'
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)
print('предсказала классы для тестовой выборки:',y_pred)
probs = logistic_regression.predict_proba(X_test) #степень уверенности в ответе, вероятность 0 и 1 (две колонки)
print('Вероятности предсказания классов 0 и 1 - две колонки для тестовой выборки: ')
print(probs)
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()
report = classification_report(y_test, y_pred)
print(report)
# ROC-анализ
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
probs = logistic_regression.predict_proba(X_test) #степень уверенности в ответе, вероятность 0 и 1 (две колонки)
# keep probabilities for the positive outcome only
probs = probs[:, 1] # отбираем вероятности для 1
# calculate AUC
my_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % my_auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.title('Логистическая регрессия. ROC from Komarov')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# show the plot
plt.show()

######################################################### Задание 2 ####################################################
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
# Шаг 2. Получите данные
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
df = pd.read_csv('DATA/diabetes.csv')  # исходный файл - обучающая выборка + классы
print("Фрагмент обучающих данных")
print(df.head(10))
feature_cols = ['Pregnancies','Glucose','Insulin','BMI','Age']
X = df[feature_cols] # Features
y = df.Outcome # Target variable
#код разделит набор данных на 70% данных обучения и 30% данных тестирования.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# Шаг 3. Создайте модель и обучите ее
model = LogisticRegression(solver='lbfgs', random_state=0) # алгоритм оптимизации, решатель (solver)='liblinear' 'newton-cg ' или 'lbfgs'
# c - Величина, обратная коэффициенту регуляризации λ,  по умолчанию 1.0.чем меньше значение, тем сильнее регуляризация.
model.fit(X_train, y_train)
# Шаг 4. Оцените модель
p_pred = model.predict_proba(X_test)
y_pred = model.predict(X_test)
#conf_m = confusion_matrix(y_test, y_pred)
#print (conf_m)
report = classification_report(y_test, y_pred)
print(report)
print (metrics.accuracy_score(y_test, y_pred))
print('предсказала классы для тестовой выборки:',y_pred)
# ROC-анализ
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
probs = model.predict_proba(X_test) #степень уверенности в ответе, вероятность 0 и 1 (две колонки)
print('Вероятности предсказания классов 0 и 1 - две колонки для тестовой выборки: ')
print(probs)
# keep probabilities for the positive outcome only
probs = probs[:, 1] # отбираем вероятности для 1
# calculate AUC
my_auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % my_auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.title('Логистическая регрессия. ROC from Komarov')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# show the plot
plt.show()

######################################################## Метод опорных векторов ########################################
import pandas as pd
from sklearn.model_selection import train_test_split # pip install scikit-learn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn import svm   #Подключили машинное обучение метод опорных векторов
# Загрузка и просмотр данных
# Шаг 2. Получите данные
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
df = pd.read_csv('DATA/diabetes.csv')  # исходный файл - обучающая выборка + классы
print("Фрагмент обучающих данных")
print(df.head(10))
feature_cols = ['Pregnancies','Glucose','Insulin','BMI','Age']
X = df[feature_cols] # Features
y = df.Outcome # Target variable
#код разделит набор данных на 70% данных обучения и 30% данных тестирования.
print(df)
# Создание корреляционной матрицы
#Корреляционная матрица показывает, какие параметры будут влиять на результат.
correlation = df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between different fearures')
plt.show()
#Кросс валидация
# Для избежания проблем с переобучением разделим наш набор данных:
# когда вы используете 42, вы всегда получите один и тот же результат при первом выполнении разделения.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 42,0,21
print(y_train)
print (X_train)
#Прогноз
model = svm.SVC(kernel='poly')  #SVR - класс для регрессии, SVC - для классификации
model.fit(X_train, y_train)
# Точность предсказания
score = model.score(X_test, y_test)
print('Точность предсказания составила: ', score)
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
# Прогноз 1 человека
x_test = np.array([[1,66,29,26.6,31]])  # это КЛИЕНТ
predicted = model.predict(x_test)
print('Прогноз диабета у КЛИЕНТА =', predicted)



from sklearn import svm, datasets
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier  # Один против остальных
from sklearn import metrics
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=30)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=42))
#classifier = OneVsRestClassifier(LinearSVC(random_state=0))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
#Оценки качества
y_pred = classifier.predict(X_test)
score_ = classifier.score(X_test, y_test) #Общая точность классификации accuracy_score
# Compare predicted class labels with actual class labels
n_predicted_correctly = np.sum(y_pred == y_test)
print("размер   тестовой выборки:",y_test.size)
print("из них корректных:",n_predicted_correctly)
print ("Результаты оценки качества модели прогнозирования:")
print ("Общая точность классификации - accuracy_score =", score_)
print ("По классам:  Точность -  Полнота - F-мера -  Кол-во")
print (metrics.classification_report(y_test, y_pred))
fpr = dict()
tpr = dict()
roc_auc = dict()
lw = 2
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = (['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()


from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier # Один против остальных
from sklearn import metrics
print("КЛАССИФИКАЦИЯ ОТ Комарова - шесть КЛАССов")
# 4.   Информация об атрибутах:
# 1. Идентификационный номер: от 1 до 214
# 2. RI: показатель преломления света
# 3. Na: натрий (единица измерения: массовые проценты в соответствующем оксиде, как и атрибуты 4-10)
# 4. Mg: магний
# 5. Al: алюминий
# 6. Si: Кремний
# 7. K: Калий
# 8. Ca: Кальций
# 9. Ba: Барий
# 10. Fe: Железо
# 11. Тип стекла: (класс)
#1.    оконное стекло (термополированное)
# 2. оконное стекло (нетермополированное)
# 3. автомобильное стекло (термополированное)
# 4. автомобильное стекло (нетермополированное) – этот тип отсутствует в данных
# 5. стекло для контейнеров
# 6. посудное
# 7. фары
data = pd.read_csv('DATA/glass.csv', sep=',', header=None)
y = np.array(data[10]) # последний столбец - классы для обучения
del data[10]   # Убрали из основной выборки. Там остались только признаки
X = np.array(data)
# Binarize the output
classes = [1,2,3,5,6,7]
y = label_binarize(y, classes= classes)
n_classes = y.shape[1]
print   ('Число классов:', n_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=0))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
#Оценки качества
y_pred = classifier.predict(X_test)
# Compare predicted class labels with actual class labels
n_predicted_correctly = np.sum(y_pred == y_test)
print("размер   тестовой выборки:",y_test.size)
print("из них корректных:",n_predicted_correctly)
print("Результаты оценки качества модели прогнозирования:")
print("Общая точность классификации - accuracy_score =", metrics.accuracy_score(y_test, y_pred))
print("По классам:  Точность -  Полнота - F-мера -  Кол-во")
print(metrics.classification_report(y_test, y_pred))
#Вывод ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
lw = 2
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = (['blue', 'red', 'green', 'yellow', 'gray', 'lightblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()



