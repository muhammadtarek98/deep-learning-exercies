import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
def normalize(image , label):
  image=tf.cast(image,tf.float32)
  image/=255
  return image,label

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
traindata,testdata=dataset['train'],dataset['test']
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

traindata=traindata.map(normalize)
testdata=testdata.map(normalize)

traindata=traindata.cache()
testdata=testdata.cache()

model=tf.keras.Sequential([
                           tf.keras.layers.Flatten(input_shape=(28, 28, 1)),

                           tf.keras.layers.Dense(128, activation=tf.nn.relu),

                           tf.keras.layers.Dense(128,activation=tf.nn.relu),

                           tf.keras.layers.Dense(10,activation=tf.nn.sigmoid)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

BATCH_SIZE = 32
traindata=traindata.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
testdata=testdata.batch(BATCH_SIZE)
model.fit(traindata,epochs=6,steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

test_loss, test_accuracy = model.evaluate(testdata, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)
