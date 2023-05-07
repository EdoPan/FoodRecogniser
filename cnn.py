# Moduli da installare: keras, tensorflow-macos
import sys
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import layers
from keras import models
from matplotlib import pyplot

# Prepara il dataset
def loadDataset():
	(trainImages, trainLabels), (testImages, testLabels) = cifar10.load_data()

	# Le immagini del dataset in questione sono quadrate con 32x32 pixel e 3 canali per il colore (presumo RGB)
	# Siamo in presenza di 50000 immagini per il training e 10000 per il testing. 
	trainImages = trainImages.reshape((50000, 32, 32, 3))
	trainImages = trainImages.astype('float32') / 255

	testImages = testImages.reshape((10000, 32, 32, 3))
	testImages = testImages.astype('float32') / 255

	trainLabels = to_categorical(trainLabels)
	testLabels = to_categorical(testLabels)
	return trainImages, trainLabels, testImages, testLabels

def defineModel():
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(128, (3, 3), activation='relu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(10, activation='softmax'))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# Grafica la Diagnostic Learning Curves
def plotDLS(history):
	# Loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')

	# Accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

	# Salva il grafico
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
 
# Evaluation of the model
def main():
	# load dataset
	trainImages, trainLabels, testImages, testLabels = loadDataset()
	model = defineModel()
	history = model.fit(trainImages, trainLabels, epochs=5, batch_size=64, validation_data=(testImages, testLabels), verbose=1)
	# evaluate model
	_, acc = model.evaluate(testImages, testLabels, verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	plotDLS(history)

if __name__ == "__main__":
    main()