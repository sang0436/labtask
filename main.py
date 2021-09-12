from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # cifar-10 dataset 가져오기

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("Train samples : ", x_train.shape, y_train.shape)  # 50000개의 32*32, 3개 채널의 train sample
print("Test samples : ", x_test.shape, y_test.shape)  # 10000개의 32*32, 3개 채널의 test sample

x_train = x_train / 255.0
x_test = x_test / 255.0  # 0~255의 값을 갖는 픽셀값을 0~1로 갖도록 조정

# CNN 모델 구현
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy : ", test_acc)
