import numpy as np
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models, layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

# cifar-10 dataset 가져오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 이미지 시각화
for i in range(1, 11):
    ax = plt.subplot(2, 5, i)
    ax.imshow(x_train[i])
    ax.set_title(np.array(class_names)[y_train[i]])
plt.show()

# valitaion set 만들기
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=1)

# sample 정보
print("Train samples : ", x_train.shape, y_train.shape)
print("Validation samples : ", x_val.shape, y_val.shape)
print("Test samples : ", x_test.shape, y_test.shape)

# 데이터 늘리기
gen = ImageDataGenerator(rotation_range=20, shear_range=0.2,
                         width_shift_range=0.2, height_shift_range=0.2,
                         horizontal_flip=True)
augment_ratio = 1.5  # 40000*1.5 = 60000개의 train sample 추가
augment_size = int(augment_ratio * x_train.shape[0])
randidx = np.random.randint(x_train.shape[0], size=augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
x_augmented, y_augmented = gen.flow(x_augmented, y_augmented, batch_size=augment_size,
                                    shuffle=False).next()
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
s = np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]

print("after augmented (train set) : ", x_train.shape, y_train.shape)

# 정규화
mean = [0, 0, 0]
std = [0, 0, 0]
new_x_train = np.ones(x_train.shape)
new_x_val = np.ones(x_val.shape)
new_x_test = np.ones(x_test.shape)
for i in range(3):
    mean[i] = np.mean(x_train[:, :, :, i])
    std[i] = np.std(x_train[:, :, :, i])
for i in range(3):
    new_x_train[:, :, :, i] = x_train[:, :, :, i] - mean[i]
    new_x_train[:, :, :, i] = new_x_train[:, :, :, i] / std[i]
    new_x_val[:, :, :, i] = x_val[:, :, :, i] - mean[i]
    new_x_val[:, :, :, i] = new_x_val[:, :, :, i] / std[i]
    new_x_test[:, :, :, i] = x_test[:, :, :, i] - mean[i]
    new_x_test[:, :, :, i] = new_x_test[:, :, :, i] / std[i]
x_train = new_x_train
x_val = new_x_val
x_test = new_x_test

# 원 핫 인코딩
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

epochs = 200

# 훈련 모델
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu',
                        input_shape=(32, 32, 3)))
model.add(BatchNormalization())

model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(BatchNormalization())

model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.001), activation='softmax'))

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_data=(x_val, y_val))
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy : ", test_acc)  # Test accuracy : 0.8187

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

# 훈련 과정 시각화 (정확도)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='blue', linestyle='solid')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='blue', linestyle='dashed')
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

# 훈련 과정 시각화 (손실)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='red', linestyle='solid')
plt.plot(epochs_range, val_loss, label='Validation Loss', color='red', linestyle='dashed')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# test
plt.rcParams["figure.figsize"] = (2, 2)
for i in range(10):
    output = model.predict(x_test[i].reshape(1, 32, 32, 3))
    plt.imshow(x_test[i].reshape(32, 32, 3))
    print("예측 : " + class_names[np.argmax(output)] + "/ 정답 : " + class_names[np.argmax(y_test[i])])
    plt.show()
