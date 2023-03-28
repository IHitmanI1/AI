import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import scipy
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# Создаем последовательную сверточную модель 2D
model = Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        #layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(), #сглаживает вывод, полученный от предыд. слоя
        layers.Dropout(0.5),   # нормализация изо
        #layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()
# Compile the model
batch_size = 128
epochs = 15
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Обучаем сеть
my_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
# Оцениваем качество обучения сети на тестовых данных
scores = model.evaluate(x_test, y_test, verbose=0)
#print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
print('Test accuracy:', scores[1]) #Test accuracy: 0.9904
print("Test loss:", scores[0])
print('Завершили тренировку модели')
# plotting the metrics
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Обучение CNN')
plt.plot(my_history.history['loss'])
plt.show()
# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(my_history.history['accuracy'])
plt.plot(my_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
# второй график
plt.subplot(2,1,2)
plt.plot(my_history.history['loss'])
plt.plot(my_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
#https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/
#Save the model
filepath = './saved_model2D'
save_model(model, filepath)

################################### Задание 3 ################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
# Загружаем данные
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
#Load the model
filepath = './saved_model2D'
# Load the model
model = load_model(filepath, compile = True)
# A few random samples
use_samples = [4, 32, 1231, 27518]  # Можно выбирать любые щт 1 до 40000
samples_to_predict = []
# Generate plots for samples
for sample in use_samples:
  # Generate a plot
  reshaped_image = x_train[sample].reshape((28, 28))
  plt.imshow(reshaped_image)
  plt.show()
  # Add sample to array for prediction
  samples_to_predict.append(x_train[sample])  # Распознавание

# Convert into Numpy array
samples_to_predict = np.array(samples_to_predict)
print(samples_to_predict.shape)

# Generate predictions for samples
predictions = model.predict(samples_to_predict)
print(predictions)

# Generate arg maxes for predictions
classes = np.argmax(predictions, axis = 1)
print('Предсказала:',classes)
#============
import cv2 as cv
# размер изо = 28x28 пикс
image = cv.imread('pretrained_networks/cat.jpg', cv.IMREAD_GRAYSCALE)  #pic 3 - хорошо
#Вывод моего изо
image = cv.resize(image, (28, 28))
plt.imshow(image)
plt.show()
# Нормализация рисунка
image = image.astype('float32')/ 255 - 0.5
image = image.reshape(1, 28, 28, 1 )
# Make sure images have shape (28, 28, 1)
#image = np.expand_dims(image, -1)
# Generate predictions for samples
predictions = model.predict(image)
print(predictions)
# Generate arg maxes for predictions
classes = np.argmax(predictions, axis = 1)
print('Предсказала: рисунок - это ', classes)

################################## Задание 4 ##################################
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
# Создаем модель с архитектурой VGG16 и загружаем веса, обученные
# на наборе данных ImageNet
model = VGG16(weights='imagenet')
# Загружаем изображение для распознавания, преобразовываем его в массив
# numpy и выполняем предварительную обработку
img_path = 'pretrained_networks/plane.jpg'  #'ship.jpg' 'cat.jpg'  'plane.jpg
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
# Запускаем распознавание объекта на изображении
preds = model.predict(x)
# Печатаем три класса объекта с самой высокой вероятностью
print('Результаты распознавания:', decode_predictions(preds, top=3)[0])

from tensorflow.keras.applications.vgg16 import decode_predictions
# convert the probabilities to class labels
label = decode_predictions(preds)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))


################################ Задание 6 ########################################
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
model = ResNet50(weights='imagenet')
img_path = 'pretrained_networks/plane.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
label = decode_predictions(preds)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))



###########################################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_pic_gen = ImageDataGenerator(rescale=1./255,
                                   fill_mode='nearest',
                                   horizontal_flip=True,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.5,
                                   shear_range=0.5,
                                   zoom_range=0.2)
test_pic_gen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
train_flow = train_pic_gen.flow_from_directory(directory=r"./train_data/",target_size=(128,128),batch_size = 32,class_mode ='binary' )
print(train_flow.class_indices)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Dropout,BatchNormalization
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters=164,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
#model.add(Conv2D(filters=164,kernel_size=(3,3),activation='relu'))
#model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
#steps_per_epoch=100,epochs= 20
his = model.fit(train_flow, epochs = 5, validation_data=train_flow)
model.save('./model_1.h5')
#Результаты обучения
import matplotlib.pyplot as plt
# Оцениваем качество обучения сети на тестовых данных
scores = model.evaluate(train_flow, verbose=0)
print('Test accuracy:', scores[1]) #Test accuracy: 0.7
print("Test loss:", scores[0])
print('Завершили тренировку модели')
# plotting the metrics
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Обучение CNN')
plt.plot(his.history['loss'])
plt.show()


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from tensorflow.keras.models import load_model
import numpy as np
pre_x = cv2.imread('./pretrained_networks/cat.jpg')  # cat.jpg,  11.jpg - кошка   31.jpg, 2.jpg - псы
pre_x1 = mpimg.imread('./pretrained_networks/cat.jpg')
cv2.imshow('show',pre_x1)
cv2.waitKey()
plt.imshow(pre_x)
pre_x = cv2.resize(pre_x,(128,128))# изменить размер во время тренировки
pre_x = cv2.cvtColor(pre_x, cv2.COLOR_BGR2RGB)# Модификация цветового макета
pre_x = pre_x/255
pre_x = np.expand_dims(pre_x,axis=0)# Вручную добавьте одно измерение, чтобы сеть могла его прочитать.
model_3 = load_model('./model_1.h5')
pre_y = model_3.predict(pre_x)
print('Вероятность собаки = ',pre_y [0][0])  # < 0.5 - кошка,  >0.5 - собачка
if pre_y[0][0] < 0.5:
    print('Это кошка')
else:
    print('это собака')
