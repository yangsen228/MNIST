from keras.datasets import mnist
from keras.utils import to_categorical

# Prepare training data
train_X, train_y = mnist.load_data()[0]
train_X = train_X.reshape(-1, 28, 28, 1)
train_X = train_X.astype('float32')
train_X /= 255
train_y = to_categorical(train_y, 10)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

# Build model
model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', input_shape=[28, 28, 1]))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss=categorical_crossentropy,
             optimizer=Adadelta(),
             metrics=['accuracy'])

# Train model
batch_size = 100
epochs = 8
model.fit(train_X, train_y,
         batch_size=batch_size,
         epochs=epochs)

# Evaluate model
test_X, test_y = mnist.load_data()[1]
test_X = test_X.reshape(-1, 28, 28, 1)
test_X = test_X.astype('float32')
test_X /= 255
test_y = to_categorical(test_y, 10)
loss, accuracy = model.evaluate(test_X, test_y, verbose=1)
print('loss:%.4f accuracy:%.4f' %(loss, accuracy))

# Output model
import h5py
from keras.models import model_from_json
json_string = model.to_json()
open('mnist_model_architecture.json', 'w').write(json_string)
model.save_weights('mnist_model_weights.h5')
