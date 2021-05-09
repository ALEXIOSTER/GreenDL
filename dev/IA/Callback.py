import tensorflow as tf
import time
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
    def on_epoch_begin(self, epoch, logs=None):
        self.timetaken = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append((epoch,time.time() - self.timetaken))

