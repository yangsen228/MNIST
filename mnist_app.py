from keras.models import model_from_json

model = model_from_json(open('mnist_model_architecture.json').read())
model.load_weights('mnist_model_weights.h5')

from keras.datasets import mnist
import cv2

# test_X.shape = (10000,28,28)
test_X, test_y = mnist.load_data()[1]

num = 999
for i in range(20):
	# test_X_sample.shape = (28,28)
	test_X_sample = test_X[i,:,:]

	test_X_sample_preds = test_X_sample.reshape(-1, 28, 28, 1)
	preds = model.predict(test_X_sample_preds)
	for i, prob in enumerate(preds[0]):
		if prob==1:
			num = i
	cv2.putText(test_X_sample, str(num), (1,8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
	cv2.imshow('test', test_X_sample)
	cv2.waitKey(1000)

