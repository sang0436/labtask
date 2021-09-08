from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# CIFAR-10 dataset 다운로드

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 데이터 정보
print("Training data")
print("Number of examples : ", train_images.shape[0])
print("Number of channels : ", train_images.shape[3])
print("Image size : ", train_images.shape[1], train_images.shape[2])
print("-----")
print("Test data")
print("Number of examples : ", test_images.shape[0])
print("Number of channels : ", test_images.shape[3])
print("Image size : ", test_images.shape[1], test_images.shape[2])
print("-----")

print("mean before normalization : ", np.mean(train_images))
print("std before normalization : ", np.std(train_images))

mean = [0, 0, 0]
std = [0, 0, 0]
new_train_images = np.ones(train_images.shape)
new_test_images = np.ones(test_images.shape)

for i in range(3):  # 채널이 3개이므로 각 채널마다의 평균, 표준편차 (train set 만)
    mean[i] = np.mean(train_images[:, :, :, i])
    std[i] = np.std(train_images[:, :, :, i])

for i in range(3):  # 정규화 작업 (train set, test set 모두)
    new_train_images[:, :, :, i] = train_images[:, :, :, i] - mean[i]
    new_train_images[:, :, :, i] = new_train_images[:, :, :, i] / std[i]
    new_test_images[:, :, :, i] = test_images[:, :, :, i] - mean[i]
    new_test_images[:, :, :, i] = new_test_images[:, :, :, i] / std[i]

train_images = new_train_images
test_images = new_test_images

print("-----")
print("mean after normalization : ", np.mean(train_images))
print("std after normalization : ", np.std(train_images))




