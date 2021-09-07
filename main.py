from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
# 모델을 만들 때 keras sequential API 사용
import matplotlib.pyplot as plt
# 이미지를 표현하기 위해 점들을 찍을 것

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# CIFAR-10 dataset 다운로드

train_images, test_images = train_images / 255.0, test_images / 255.0
# RGB 픽셀값을 0~1 사이 정규화

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

