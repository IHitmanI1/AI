import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from tensorflow.keras.models import load_model
import numpy as np
pre_x = cv2.imread('./pretrained_networks/cat1.jpg')  # cat.jpg,  11.jpg - кошка   31.jpg, 2.jpg - псы
pre_x1 = mpimg.imread('./pretrained_networks/cat1.jpg')
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