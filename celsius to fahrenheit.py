import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plot
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)
l0=tf.keras.layers.Dense(units=1 , input_shape=[1])
model=tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))
history=model.fit(celsius_q,fahrenheit_a,epochs=500,verbose=False)
plot.xlabel('iterations')
plot.ylabel('loss')
plot.plot(history.history['loss'])
print(model.predict([100.0]))
print("w= {} ".format(l0.get_weights()))
